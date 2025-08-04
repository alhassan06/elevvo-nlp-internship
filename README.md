# 📰 Fake News Detection

This project is part of an NLP internship task. It classifies news articles as **real or fake** using machine learning and text preprocessing techniques.

---

## 📂 Dataset

- Source: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Files used: `True.csv`, `Fake.csv`

---

## 🧠 Task Overview

- Combine `True.csv` and `Fake.csv` into a single dataset
- Preprocess the `title` and `text` columns:
  - Lowercase
  - Remove punctuation and stopwords
  - Lemmatize tokens
- Convert text to numerical features using **TF-IDF**
- Train a **Logistic Regression** or **SVM** model
- Evaluate using:
  - **Accuracy**
  - **F1-score**
  - **Classification report**

---

## 🧰 Tools & Libraries

- Python 3.x
- `pandas`, `numpy`
- `nltk`, `spaCy`
- `scikit-learn`
- `matplotlib`, `seaborn`

---

## 🗂️ Folder Structure

fake-news-detection/
├── data/ # Contains True.csv and Fake.csv
├── scripts/ # Contains main.py
├── notebook/ # Contains final Jupyter notebook
├── nlp_env/ # Python virtual environment
├── requirements.txt # Dependencies
└── README.md

yaml
Copy code

---

## ▶️ How to Run

1. **Activate virtual environment**:

    ```bash
    .\nlp_env\Scripts\activate
    ```

2. **Run the main script**:

    ```bash
    python scripts/main.py
    ```

---

## ✅ Results

- **TF-IDF feature size**: 5000
- **Train samples**: 35,918  
- **Test samples**: 8,980  
- **Accuracy**: ~98.88%

---

## 📌 Notes

- The notebook version is available in the `notebook/` folder for easy experimentation and visualization.
- Model performance may slightly vary depending on preprocessing choices.

---
