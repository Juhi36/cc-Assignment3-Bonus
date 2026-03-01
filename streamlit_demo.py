
import pandas as pd
import streamlit as st
from sentiment_analysis import SentimentAnalysis

# Cache the model so it is not created on every button press
@st.cache_data
def do_analysis_model():
    st.progress(0, "Wait for the model to get ready")
    sa = SentimentAnalysis()
    return sa


# Analysis is done on a st fragment
@st.fragment
def do_analysis(sa: SentimentAnalysis):
    analyzed = False
    prediction = ""
    if st.button("Analyze!"):
        prediction = sa.Predict_sentence(inputStr)
        analyzed = True

    if analyzed:
        # Show result when sentence is analyzed
        st.markdown(f"Your input was analyzed to be: {prediction}")


sa = do_analysis_model()
st.title("Sentiment analysis using Streamlit!")
inputStr = st.text_input("Input your sentence")

do_analysis(sa)
