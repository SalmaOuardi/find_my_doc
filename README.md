# 🩺 FindMyDoc

**FindMyDoc** is a modular, AI-powered assistant that helps users find relevant doctors or medical services based on free-form user queries. It supports both classic machine learning and transformer-based NLP models, with an interactive CLI, confidence scores, and CI/CD-ready structure.

---

## 🚀 Features

- **Intent Classification**: Categorizes queries as `COR` (Correct), `OOC` (Out of Context), or `INC` (Inappropriate).
- **Model Options**:
  - `sklearn` (TF-IDF + Logistic Regression)
  - `roberta-base`
  - `microsoft/deberta-v3-small`
- **Interactive CLI**: Train and test models directly in your terminal.
- **Confidence Scores**: Displays intent along with prediction confidence.
- **Unit Testing**: Pytest coverage for intent predictions and edge cases.
- **CI/CD Ready**: Includes GitHub Actions workflow for automated testing.
- **MLOps Friendly**: Modular, production-grade architecture.

---

## 🧱 Project Structure

```
FindMyDoc/
├── src/
│   └── find_my_doc/
│       ├── __init__.py
│       ├── intent_classifier.py
│       └── utils.py
├── data/
│   └── intent_classification_dataset.csv
├── models/                  # Saved models (ignored in git)
├── tests/
│   └── test_intent.py
├── main.py                  # CLI for training and prediction
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/FindMyDoc.git
cd FindMyDoc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Usage

### 📌 Train a Model

```bash
python main.py
```

Then choose: `sklearn`, `roberta`, or `deberta` when prompted.

### 💬 Run Predictions Interactively

Once trained, test it right in the terminal:

```
💬 Enter a user query: I need a neurologist in Boston
🔍 Predicted intent: COR (92.5%)
```

---

## ✅ Run Tests

```bash
pytest
```

---

## 📊 Model Comparison

| Model     | Speed   | Accuracy | Notes                        |
|-----------|---------|----------|------------------------------|
| sklearn   | ⚡ Fast  | 🧠 Basic | Good for quick testing       |
| roberta   | 🚀 Mid  | 🔍 Better | Strong general performance   |
| deberta   | 🐢 Slow | 🎯 Best   | Most accurate, context-aware|

---

## 📍 Roadmap

- [x] Intent classification pipeline
- [x] Transformer model integration
- [x] CLI with confidence score output
- [x] Unit test coverage with pytest
- [ ] NER + entity linking module
- [ ] Streamlit or FastAPI UI
- [ ] Hugging Face Spaces / Render deployment

---

## 👩‍💻 Author

**Salma Ouardi**  

---

## 📄 License

MIT License
