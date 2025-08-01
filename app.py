import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words and len(word) > 2])

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article or headline, and this model will predict whether it's **Fake** or **Real**.")

user_input = st.text_area("Paste your news text here 👇", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("❌ This news is **FAKE**.")
