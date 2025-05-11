from intent_classifier import evaluate_intent_classifier
import pandas as pd

results = []
for model_type in ["sklearn", "roberta", "deberta"]:
    print(f"Evaluating {model_type}...")
    result = evaluate_intent_classifier("../../data/Intent_toy_dataset.csv", model_type=model_type)
    results.append(result)

df = pd.DataFrame(results)
print(df.to_markdown(index=False))  # Pretty terminal output
df.to_csv("results/model_comparison.csv", index=False)  # Save as CSV
