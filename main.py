import streamlit as st
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    st.title("Text Sentiment Analysis App")

    # Text input area
    user_input = st.text_area("Enter text and press Enter:", key="user_input", height=100)

if __name__ == "__main__":
    main()
