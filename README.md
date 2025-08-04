# News Classification Task

This project performs **news article classification** using **TF-IDF vectorization and Logistic Regression** to predict the category of news articles based on their title and description.

It was developed as part of the **Elevvo Pathways NLP learning track** for practical machine learning training.

---

## 📂 Folder Structure

news-classification-task/
├── data/
│ └── train.csv
├── models/
│ ├── news_classifier_logreg.pkl
│ └── tfidf_vectorizer.pkl
├── notebooks/
│ └── news_classification_task.ipynb
├── scripts/
│ └── news_classification_task.py 
└── README.md


yaml
Copy code

---

## 🛠️ Requirements

Ensure you have **Python 3.13.x** and the following packages installed in your environment:

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- spacy
- scikit-learn
- joblib

If needed, install using:
```bash
pip install pandas numpy matplotlib seaborn nltk spacy scikit-learn joblib
python -m spacy download en_core_web_sm
🚀 How to Run
1️⃣ Clone or download the repository to your local machine.

2️⃣ Ensure your train.csv is placed inside the data/ folder:

bash
Copy code
news-classification-task/data/train.csv
3️⃣ Activate your virtual environment:

bash
Copy code
cd path\to\news-classification-task
.\nlp_env\Scripts\activate
4️⃣ Open the notebook in VS Code:

Open notebooks/news_classification_task.ipynb.

Ensure the kernel is set to nlp_env.

Run all cells.

5️⃣ The notebook will:
✅ Preprocess your data (lowercasing, cleaning, lemmatization).
✅ Vectorize text using TF-IDF.
✅ Train Logistic Regression on your data.
✅ Evaluate and print accuracy and classification report.
✅ Display the confusion matrix.
✅ Save the trained model and vectorizer for future use.

📊 Results
Using train.csv with your current workflow, you achieved:

Accuracy: 89.6% on the test set.

A clear confusion matrix showing the model's performance across the four news categories.

💾 Outputs
After running, the following will be saved automatically inside models/:

news_classifier_logreg.pkl – your trained Logistic Regression model.

tfidf_vectorizer.pkl – your trained TF-IDF vectorizer for reuse.

You can use these for future inference without retraining.

✅ Status
✅ Project implemented, trained, evaluated, and working correctly in VS Code + Jupyter Notebook.
✅ Folder organized for clarity and submission.
✅ Ready for learning demonstration and future fine-tuning.

🤝 Contribution
This project is part of Elevvo Pathways NLP Track under your practical exercises. If you wish to extend it, consider:

Trying different models (SVM, Random Forest, Naive Bayes).

Hyperparameter tuning for higher accuracy.

Testing on additional news data for robustness.