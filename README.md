# twitter-sentiment-analysis
Twitter Sentiment Analysis using NLP &amp; Machine Learning
## 🔗 Live Demo
Streamlit App:  
https://twitter-sentiment-analysis-tfundtrjyueokz9yudjcmd.streamlit.app/

# Twitter Sentiment Analysis using NLP & Machine Learning

This project performs **sentiment analysis on Twitter tweets** using **Natural Language Processing (NLP)** and **Machine Learning**.  
The system classifies tweets as **Positive 😀 or Negative 😞** and is deployed as an interactive **Streamlit web application**.

---

## 📌 Project Overview
Social media platforms like Twitter generate massive amounts of unstructured text data.  
Manually analyzing sentiment is not scalable, so this project automates the process using NLP and ML techniques to extract meaningful insights from tweets.

---

## 🚀 Features
- Tweet text cleaning and preprocessing
- TF-IDF feature extraction
- Multiple ML models trained and evaluated
- Best model selected based on performance
- Real-time sentiment prediction using Streamlit UI

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, Joblib  
- **NLP Technique:** TF-IDF Vectorization  
- **ML Models:** Naive Bayes, SVM, Logistic Regression  
- **Deployment:** Streamlit  

---

## 📂 Project Files
- `FinalCP2.ipynb` → Data preprocessing, EDA, model training & evaluation  
- `app.py` → Streamlit web application  
- `sentiment_model.pkl` → Trained Logistic Regression model  
- `tfidf_vectorizer.pkl` → Saved TF-IDF vectorizer  

---

## 📊 Dataset
The full dataset is **not included** in this repository due to size limitations.

- **Dataset Name:** Sentiment140  
- **Source:** Kaggle – Sentiment140 Twitter Dataset  
- **Labels:**  
  - `0` → Negative sentiment  
  - `1` → Positive sentiment  

A sampled subset of **100,000 tweets** was used for training and evaluation.

---

## 📈 Model Performance
| Model | Accuracy |
|------|----------|
| Naive Bayes | ~76% |
| SVM | ~78% |
| Logistic Regression | **~79% (Best)** |

Logistic Regression was selected due to its balance of accuracy, interpretability, and fast inference.

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

