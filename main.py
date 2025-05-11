from src.find_my_doc.intent_classifier import train_intent_classifier, predict_intent

def main():
    print("\n🚀 FindMyDoc - Intent Classification CLI")
    mode = input("\n🧠 Select mode (train / infer): ").strip().lower()

    if mode not in {"train", "infer"}:
        print("❌ Invalid mode. Choose 'train' or 'infer'.")
        return

    print("\n📦 Available models:")
    print(" 1. sklearn (TF-IDF + Logistic Regression)")
    print(" 2. roberta (Hugging Face transformer)")
    print(" 3. deberta (Hugging Face transformer)")

    model_choice = input("\n👉 Which model would you like to use? (sklearn / roberta / deberta): ").strip().lower()

    if model_choice not in {"sklearn", "roberta", "deberta"}:
        print("❌ Invalid choice. Please enter 'sklearn', 'roberta', or 'deberta'.")
        return

    if mode == "train":
        train_intent_classifier("data/Intent_toy_dataset.csv", model_type=model_choice)
        print(f"\n✅ Training complete. You can now use the {model_choice} model for inference.\n")

    print(f"\n🤖 Entering inference mode using the {model_choice} model...\n")

    while True:
        query = input("💬 Enter a user query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        intent, confidence = predict_intent(query, model_type=model_choice)
        print(f"🔍 Predicted intent: {intent} ({confidence}%)\n")

if __name__ == "__main__":
    main()
