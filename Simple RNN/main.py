import numpy as np
import tensorflow as tf
from pathlib import Path
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
# Adjust index to align with Keras reserved tokens (PAD/START/UNK/UNUSED).
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

model_path = Path(__file__).with_name("imdb_rnn_model.h5")
model = load_model(str(model_path))

def decode_review(encoded_review):
    # Skip padding tokens to avoid a wall of placeholders.
    tokens = [i for i in encoded_review if i != 0]
    return ' '.join([reverse_word_index.get(i, "<UNK>") for i in tokens])

def preprocess_text(text):
    # Keep words and apostrophes; drop punctuation to reduce OOV.
    words = re.findall(r"[a-z']+", text.lower())
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

###Prediction function
def predict_review(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)

    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    return sentiment,prediction[0][0]

import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as postive or negative')

user_input = st.text_area('Enter your review here:')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    score = prediction[0][0]
    st.write(f'Predicted Sentiment: {sentiment} (Score: {score:.4f})')
    st.write(f'Decoded Review: {decode_review(preprocess_input[0])}')
else:
    st.write('Please enter a review to classify')