import streamlit as st
import joblib
import re
import string

# ---------------------------------
# 1. Configure Page
# ---------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# ---------------------------------
# 2. Load model + vectorizer
# ---------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_files()

# ---------------------------------
# 3. Cleaning function
# (must be SAME as used in training)
# ---------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    if pred == 1:
        return "Positive 😀", cleaned
    else:
        return "Negative 😞", cleaned

# ---------------------------------
# 4. Sidebar
# ---------------------------------
st.sidebar.title("ℹ️ About the App")
st.sidebar.write("""
This app uses **Machine Learning + NLP** to classify tweets as:
- Positive 😀
- Negative 😞

**Model used:** Logistic Regression  
**Features:** TF-IDF (5000)  
**Dataset:** Sentiment140  
""")

st.sidebar.markdown("---")
st.sidebar.write("Created by **Nikhil Dange Patil**")

# ---------------------------------
# 5. Main Layout
# ---------------------------------
st.markdown("<h1 style='text-align:center;'>🐦 Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Analyze the sentiment of any tweet in real time.</p>", unsafe_allow_html=True)

left, right = st.columns([2, 1])

# ---------------------------------
# 5A. Left Section – Input & Prediction
# ---------------------------------
with left:
    st.subheader("🔹 Enter Tweet")

    user_text = st.text_area("Write your tweet here:", height=150)

    analyze = st.button("🔍 Analyze Sentiment")

    if analyze:
        if user_text.strip() == "":
            st.warning("Please enter a tweet!")
        else:
            result, cleaned_text = predict_sentiment(user_text)

            st.subheader("🔰 Prediction")
            if "Positive" in result:
                st.success(result)
            else:
                st.error(result)

            with st.expander("Show cleaned tweet"):
                st.code(cleaned_text)

# ---------------------------------
# 5B. Right Section – Samples & Model Info
# ---------------------------------
with right:
    st.subheader("📊 Model Summary")
    st.write("""
    **Algorithm:** Logistic Regression  
    **Accuracy:** ~79%  
    **Training Data:** 100,000 tweets  
    """)

    st.markdown("---")
    st.subheader("✨ Try Sample Tweets")

    positive_text = "I absolutely love this product! Amazing quality."
    negative_text = "This is the worst experience ever, I am disappointed."
    neutral_text = "Not bad, could be better."

    if st.button("👍 Positive Example"):
        st.info(positive_text)
        st.write("Prediction:", predict_sentiment(positive_text)[0])

    if st.button("👎 Negative Example"):
        st.info(negative_text)
        st.write("Prediction:", predict_sentiment(negative_text)[0])

    if st.button("😐 Neutral-ish Example"):
        st.info(neutral_text)
        st.write("Prediction:", predict_sentiment(neutral_text)[0])

st.markdown("---")
st.caption("NLP Model: Logistic Regression + TF-IDF trained on Sentiment140 (100k tweets).")
