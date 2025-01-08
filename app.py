import streamlit as st 
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('next_word_lstm.keras')
model1 = load_model('next_word_gru.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Next word prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1).item()
    return tokenizer.index_word.get(predicted_word_index, "Unknown")

# Streamlit App
st.title('Next Word Prediction using LSTM')

# Unique key for LSTM input box
input_text_lstm = st.text_input("Enter the Sequence of Words for LSTM", "To be or not to", key="lstm_input")

if st.button("Predict Next Word (LSTM)", key="lstm_button"):
    max_sequence_len = model.input_shape[1] + 1
    next_word_lstm = predict_next_word(model, tokenizer, input_text_lstm, max_sequence_len)
    st.write(f"Next Word (LSTM): {next_word_lstm}")

st.title('Next Word Prediction using GRU')

# Unique key for GRU input box
input_text_gru = st.text_input("Enter the Sequence of Words for GRU", "To be or not to", key="gru_input")

if st.button("Predict Next Word (GRU)", key="gru_button"):
    max_sequence_len = model1.input_shape[1] + 1
    next_word_gru = predict_next_word(model1, tokenizer, input_text_gru, max_sequence_len)
    st.write(f"Next Word (GRU): {next_word_gru}")
