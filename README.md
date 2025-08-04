# Sentiment Analysis on IMDB Product Reviews

This project performs **sentiment analysis on product reviews** using **Logistic Regression with TF-IDF vectorization**.

The goal is to classify reviews as **positive or negative** and evaluate the model's performance.

---

## 📂 Project Structure

- `data/` - Contains `imdb_dataset.csv` used for training.
- `notebook/` - Contains the main notebook `sentiment_analysis.ipynb`.
- `outputs/` - Stores generated outputs such as confusion matrix images and sample predictions.
- `README.md` - Project overview and instructions.
- `requirements.txt` - Python dependencies.
- `nlp_env`

---

## ⚙️ Setup Instructions

1️⃣ Clone this repository or download it as a ZIP.

2️⃣ Install required dependencies:

```bash
pip install -r requirements.txt
3️⃣ Open the notebook:

bash
Copy code
jupyter notebook notebook/sentiment_analysis.ipynb
4️⃣ Run all cells to:

Preprocess the data.

Train the model.

Evaluate accuracy and view the confusion matrix.

📈 Results
Model: Logistic Regression + TF-IDF

Accuracy: ~90% on the test set

Evaluation:

Confusion matrix saved in outputs/confusion_matrix.png

Classification report printed at the end of the notebook

🚀 Future Improvements
Hyperparameter tuning with GridSearchCV.

Try SVM or deep learning models (LSTM, BERT).

Perform error analysis on misclassified samples.

Add a Streamlit app for live predictions.

🩶 Acknowledgements
Dataset: IMDB Dataset on Kaggle

Libraries used: pandas, scikit-learn, matplotlib, seaborn, nltk.

Feel free to explore the notebook to understand preprocessing, training, and evaluation workflows clearly.

yaml
Copy code
