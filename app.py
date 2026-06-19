import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. UI Setup
st.set_page_config(page_title="Email Spam Classifier", layout="centered")
st.title("📧 Email/SMS Spam Classifier")
st.write("Enter any message below to check if it's Spam or Safe.")

# 2. Model Training (Strictly Local File Format)
@st.cache_data
def load_and_train_model():
    # Humne separator ko '\t' rakha hai kyunki humne tsv data copy kiya hai
    df = pd.read_csv("spam.csv", sep='\t', header=None, names=['label', 'text'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['text']
    y = df['label']
    
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    
    model = MultinomialNB()
    model.fit(X, y)
    
    return cv, model

try:
    cv, model = load_and_train_model()
    
    # 3. User Input
    user_input = st.text_area("Paste your Email/SMS here:", height=150, placeholder="Type here...")

    if st.button("Predict / Check Status", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter some text first!")
        else:
            data = cv.transform([user_input]).toarray()
            prediction = model.predict(data)
            
            st.write("---")
            if prediction[0] == 1:
                st.error("🚨 **Alert: This looks like a SPAM message!**")
            else:
                st.success("✅ **Safe: This is a normal message (HAM).**")

except Exception as e:
    st.error(f"Error loading local file: {e}")