import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# Load model
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# Text cleaning
def clean_text(text):

    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Page settings
st.set_page_config(
    page_title="AI Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Header
st.markdown(
"""
<h1 style='color: blue; text-align: center;'>AI Fake News Detection System</h1>
<p style='text-align: center; color: grey;'>
Enter a news title and article text to check whether the news is Fake or Real..
</p>
""",
unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("Enter News Details")

title = st.text_input("News Title")

article = st.text_area("News Article", height=100)

st.divider()

# Prediction button
if st.button("Analyze News"):

    if title == "" or article == "":
        st.warning("WARNING: Please provide both title and article. ")

    else:

        content = title + " " + article
        content = clean_text(content)

        vector = vectorizer.transform([content])

        prediction = model.predict(vector)

        st.subheader("Result")

        if prediction[0] == 1:
            st.error("⚠️ This article appears to be FAKE.")
        else:
            st.success("✅ This article appears to be REAL.")

# Footer
st.markdown(
"""
<p style='text-align:center; font-size:15px; color:gray;'>
This tool uses NLP and machine learning to analyze news content.
</p>
""",
unsafe_allow_html=True
)