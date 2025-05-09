from src.medgpt.intent_classifier import train_intent_classifier, predict_intent

def main():
    print("\n📦 Available models:")
    print(" 1. sklearn (TF-IDF + Logistic Regression)")
    print(" 2. roberta (Hugging Face transformer)")
    print(" 3. deberta (Hugging Face transformer)")

    model_choice = input("\n👉 Which model would you like to train? (sklearn / roberta / deberta): ").strip().lower()

    if model_choice not in {"sklearn", "roberta", "deberta"}:
        print("❌ Invalid choice. Please enter 'sklearn', 'roberta', or 'deberta'.")
        return

    train_intent_classifier("data/Intent_toy_dataset.csv", model_type=model_choice)

    print(f"\n✅ Training complete. Now testing the {model_choice} model...\n")
    while True:
        query = input("💬 Enter a user query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        intent, confidence = predict_intent(query, model_type=model_choice)
        print(f"🔍 Predicted intent: {intent} ({confidence}%)\n")

if __name__ == "__main__":
    main()
