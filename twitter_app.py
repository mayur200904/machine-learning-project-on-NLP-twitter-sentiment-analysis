# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:26:08 2024

@author: vaibh
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# Load the saved model
model=tf.keras.models.load_model("sentiment_model.h5")

# Load the tokenizerr
tokenizer=Tokenizer(num_words=5000)
max_length=100

# Create the function to predict
def predict_sentiment(text):
    # Tokenize
    sequences=tokenizer.texts_to_sequences([text])
    padded=pad_sequences(sequences,maxlen=max_length,truncating="post")
    
    # Predict the sentiment
    prediction=model.predict(padded)
    
    return prediction

# Built the UI
st.title("Sentimental Analysis APP")

# Text input from user
user_input=st.text_area("Enter the text")

# Main processing
if st.button("Predict"):
    if user_input:
        # Call the predict function
        predict=predict_sentiment(user_input)
        
        # Determine the sentiment based on the predict
        sentiment=["Negative","Neutral","Positive"][predict.argmax()]
        
        # Display the result
        st.write(f"Sentiment: {sentiment}")
        
    else:
        st.warning("Please enter some text for analysis")