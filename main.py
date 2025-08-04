import pandas as pd
import numpy as np
import string
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

true_df=pd.read_csv("data/True.csv")
fake_df=pd.read_csv("data/Fake.csv")
true_df["label"]=1
fake_df["label"]=0
df=pd.concat([true_df, fake_df],ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(df.head())
print(f"\nTotal samples: {len(df)}")
print(df['label'].value_counts())

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
df["content"] = df["title"] + " " + df["text"]
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)
df["cleaned_content"] = df["content"].apply(preprocess)
print("\nOriginal:")
print(df["content"].iloc[0][:200])
print("\nCleaned:")
print(df["cleaned_content"].iloc[0][:200])
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df["cleaned_content"])
y=df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTF-IDF feature shape:", X.shape)
print("Train set:", X_train.shape)
print("Test set:", X_test.shape)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')