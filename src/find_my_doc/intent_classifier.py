import pandas as pd
# Standard library
import os
import time

# Data handling
import pandas as pd
import joblib

# Transformers
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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
            return tokenizer(example["text"], truncation=True, padding=True, max_length=128)

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
        pipeline = joblib.load("../../models/intent_model_sklearn.pkl")
        pred = pipeline.predict([query])[0]
        proba = pipeline.predict_proba([query])[0]
        label_index = list(pipeline.classes_).index(pred)
        confidence = round(proba[label_index] * 100, 2)
        return pred, confidence

    elif model_type in ["roberta", "deberta"]:
        model_dir = f"../../models/hf_{model_type}"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        label_map = {0: "COR", 1: "OOC", 2: "INC"}
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        label = label_map[pred_idx.item()]
        return label, round(confidence.item() * 100, 2)

def evaluate_intent_classifier(data_path, model_type="sklearn"):
    df = pd.read_csv(data_path)
    label_map = {"COR": 0, "OOC": 1, "INC": 2}
    reverse_map = {v: k for k, v in label_map.items()}
    df["label_id"] = df["intent"].map(label_map)

    texts = df["text"].tolist()
    true = df["intent"].tolist()
    true_ids = df["label_id"].tolist()

    preds = []
    start_time = time.time()

    for text in texts:
        label, _ = predict_intent(text, model_type=model_type)
        preds.append(label)

    elapsed = round(time.time() - start_time, 2)
    acc = accuracy_score(true, preds)
    report = classification_report(true, preds, output_dict=True)

    return {
        "model": model_type,
        "accuracy": round(acc, 4),
        "precision_macro": round(report["macro avg"]["precision"], 4),
        "recall_macro": round(report["macro avg"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "inference_time": elapsed,
    }