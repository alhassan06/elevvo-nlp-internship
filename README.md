# 🧠 NLP Internship Projects – Elevvo

This repository contains the NLP tasks I completed during my Elevvo Internship. Each project tackles a different real-world NLP problem, using industry-relevant datasets and techniques in Python.

---

## ✅ Completed Tasks

### 1. Question Answering with Transformers
- **Dataset:** SQuAD v1.1 (via HuggingFace)
- **Objective:** Build a system that extracts answers to questions from a given context.
- **Tools:** HuggingFace Transformers (`bert-large-uncased-whole-word-masking-finetuned-squad`)
- **Evaluation:** Exact Match (EM), F1 Score

### 2. Sentiment Analysis on Product Reviews
- **Dataset:** IMDb Reviews / Amazon Product Reviews (Kaggle)
- **Objective:** Determine if a product review is positive or negative.
- **Preprocessing:** Lowercasing, stopword removal
- **Vectorization:** TF-IDF
- **Model:** Logistic Regression
- **Evaluation:** Accuracy, Confusion Matrix

### 3. News Category Classification
- **Dataset:** News Category Dataset (e.g., HuffPost or Kaggle)
- **Objective:** Predict the category of a news article (e.g., tech, business, politics).
- **Model:** Multiclass Logistic Regression
- **Evaluation:** Accuracy, Precision, Recall

### 4. Fake News Detection
- **Dataset:** Fake and Real News Dataset (Kaggle)
- **Objective:** Classify whether a news article is real or fake.
- **Preprocessing:** Stopword removal, lemmatization
- **Vectorization:** TF-IDF
- **Model:** Logistic Regression / SVM
- **Evaluation:** Accuracy, F1 Score

---

## 🛠️ Tools & Libraries

- Python 3
- pandas, numpy
- scikit-learn
- NLTK / spaCy
- HuggingFace Transformers
- Jupyter Notebook

---
## 📌 Notes

- Each task is self-contained with its own scripts and notebooks.
- Notebooks include preprocessing, modeling, evaluation, and explanation steps.
- All models were trained and tested locally using open-source datasets.

---



