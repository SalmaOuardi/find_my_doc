import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# TODO: change the model to RoBERTa or DeBERTa.
# TODO: Create a bigger dataset with more intents and examples.

def train_intent_classifier(data_path: str, model_type="sklearn"):
    df = pd.read_csv(data_path)
    label_map = {"COR": 0, "OOC": 1, "INC": 2}
    df["label"] = df["intent"].map(label_map)

    if model_type == "sklearn":
        X, y = df["text"], df["intent"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=500))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/intent_model_sklearn.pkl")
        print("Saved sklearn model to models/intent_model_sklearn.pkl")

    elif model_type in ["roberta", "deberta"]:
        model_name = {
            "roberta": "roberta-base",
            "deberta": "microsoft/deberta-v3-small"
        }[model_type]

        df_hf = df[["text", "label"]]  # HF needs label column
        dataset = Dataset.from_pandas(df_hf)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, padding=True)

        dataset = dataset.map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        save_dir = f"models/hf_{model_type}"
        training_args = TrainingArguments(
            output_dir=save_dir,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            logging_dir=f"{save_dir}/logs",
            evaluation_strategy="no",
            save_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )

        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved {model_type} model to {save_dir}")
      
def predict_intent(query: str, model_type="sklearn"):
    if model_type == "sklearn":
        pipeline = joblib.load("models/intent_model_sklearn.pkl")
        pred = pipeline.predict([query])[0]
        proba = pipeline.predict_proba([query])[0]
        label_index = list(pipeline.classes_).index(pred)
        confidence = round(proba[label_index] * 100, 2)
        return pred, confidence

    elif model_type in ["roberta", "deberta"]:
        model_dir = f"models/hf_{model_type}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        label_map = {0: "COR", 1: "OOC", 2: "INC"}
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        label = label_map[pred_idx.item()]
        return label, round(confidence.item() * 100, 2)

