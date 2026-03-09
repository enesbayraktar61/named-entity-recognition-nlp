import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(page_title="Named Entity Recognition", page_icon="🏷️", layout="centered")

#Load model
model = tf.keras.models.load_model("ner_bilstm_model.keras")

# Load preprocessing objects
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("idx2tag.pkl", "rb") as f:
    idx2tag = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# App title
st.title("🏷️ Named Entity Recognition (NER)")
st.write("Enter a sentence and the model will identify named entities such as persons, organizations, and locations.")

# Text input
user_input = st.text_area("Enter text here:")

# Prediction button
if st.button("Predict Entities"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input by splitting on spaces
        words = user_input.split()

        # Convert words to indices
        sequence = [word2idx.get(word, word2idx["UNK"]) for word in words]

        # Pad sequence
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="post",
            value=word2idx["PAD"]
        )

        # Predict tags
        predictions = model.predict(padded_sequence, verbose=0)
        predicted_labels = np.argmax(predictions, axis=-1)[0]

        # Show results
        st.subheader("Predicted Entities")

        results = []
        for word, pred_idx in zip(words, predicted_labels[:len(words)]):
            tag = idx2tag[pred_idx]
            results.append((word, tag))

        for word, tag in results:
            st.write(f"**{word}** → {tag}")