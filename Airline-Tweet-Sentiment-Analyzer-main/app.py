import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Set page configuration
st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="wide")

# Title
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.write("This app uses a Naive Bayes model to predict the sentiment of airline tweets (Positive, Neutral, or Negative).")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Tweets.csv')
    df = df[df['airline_sentiment_confidence'] >= 0.5]
    return df

sentiment_df = load_data()

# Preprocess
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words and word not in string.punctuation]
    return ' '.join(text)

sentiment_df['clean_text'] = sentiment_df['text'].apply(preprocess_text)
sentiments = ['negative', 'neutral', 'positive']
sentiment_df['label'] = sentiment_df['airline_sentiment'].apply(lambda x: sentiments.index(x))

# Features and Labels
X = sentiment_df['clean_text']
Y = sentiment_df['label']

# Vectorization
vectorizer = CountVectorizer(max_features=5000, stop_words=['virginamerica','united'])
X_vect = vectorizer.fit_transform(X).toarray()

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X_vect, Y, test_size=0.3)
model = MultinomialNB()
model.fit(X_train, Y_train)

# Sidebar
st.sidebar.header("Enter a tweet to analyze sentiment:")
user_input = st.sidebar.text_area("Tweet", value="I love flying with Delta!")

if st.sidebar.button("Predict Sentiment"):
    cleaned_input = preprocess_text(user_input)
    input_vect = vectorizer.transform([cleaned_input]).toarray()
    prediction = model.predict(input_vect)[0]
    predicted_sentiment = sentiments[prediction]

    st.sidebar.subheader("🔍 Prediction:")
    st.sidebar.success(f"**{predicted_sentiment.capitalize()}**")

# Show Metrics
if st.checkbox("Show Model Performance on Test Data"):
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.code(classification_report(Y_test, y_pred, target_names=sentiments))
