import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')
# print(fake_df.head(5))
# print(real_df.head(5))
fake_df['label'] = 0
real_df['label'] = 1
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
# print(df.head())

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words and len(word) > 2])

df['text_clean'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_clean']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import pickle

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
