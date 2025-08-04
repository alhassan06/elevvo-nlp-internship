import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


data = pd.read_csv("train.csv")
print(data.head)
print(data.info())
print(data.isnull().sum())
data['text'] = data['Title'] + " " + data['Description']
y= data['Class Index']
X=data['text']
print (X.head())
print(y.head())
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text=text.lower()
    text=re.sub(r'[^a-z\s]', '', text)
    doc=nlp(text)
    tokens=[token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
    return " ".join(tokens)

X_clean = X.apply(preprocess_text)
print(X_clean.head())
data['clean_text'] = X_clean
vectorizer=TfidfVectorizer(max_features=4000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['Class Index']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4], yticklabels=[1,2,3,4])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
joblib.dump(model, "news_classifier_logreg.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

