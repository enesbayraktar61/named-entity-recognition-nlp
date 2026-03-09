# Named Entity Recognition (BiLSTM)

This project builds a Named Entity Recognition (NER) system to identify entities such as persons, organizations, locations, and dates in text using deep learning.

The model was trained using TensorFlow and deployed with Streamlit.

---

## Project Overview

- **Problem Type:** Named Entity Recognition (Sequence Labeling)  
- **Approach:** Deep Learning (BiLSTM)  
- **Framework:** TensorFlow / Keras  
- **Deployment:** Streamlit  

---

## Dataset

The dataset consists of token-level annotated sentences used for NER tasks.

- Each row represents a single word  
- Words are grouped into sentences  
- Entity labels include:

  - **B-per / I-per** → Person  
  - **B-org / I-org** → Organization  
  - **B-geo / I-geo** → Location  
  - **B-gpe / I-gpe** → Geo-political entity  
  - **O** → Outside (no entity)

The dataset contains over **1 million tokens** and nearly **48k sentences**, making it suitable for sequence labeling tasks.

---

## Data Preprocessing

### Sentence Preparation

- Filled missing sentence IDs  
- Reconstructed full sentences from tokens  
- Removed missing words  

### Encoding

- Converted words into numerical indices  
- Converted entity tags into numerical labels  
- Applied padding (**max_len = 104**)  

These steps ensured compatibility with deep learning models.

---

## Modeling

### Deep Learning Strategy

- **Embedding layer** for word representation  
- **Bidirectional LSTM** for contextual understanding  
- **TimeDistributed Dense layer** for token-level classification  

The model was trained for 3 epochs with validation monitoring.

---

## Results

The model achieved strong performance:

- **Training Accuracy:** ≈ 94%  
- **Validation Accuracy:** ≈ 94%  
- **Test Accuracy:** ≈ 93.8%  

Although accuracy can appear high due to many non-entity tokens ("O"), qualitative evaluation shows strong entity recognition performance.

---

## Deployment

The trained model was saved in `.keras` format along with preprocessing mappings.

The Streamlit application allows users to:

- Enter text  
- Detect named entities  
- View token-level predictions  

---

## Conclusion

This project demonstrates an end-to-end NLP workflow for sequence labeling tasks.

The BiLSTM model successfully captures contextual relationships between words and accurately detects named entities. Proper preprocessing and structured experimentation were essential for achieving strong results.

---

## How to Run Locally

git clone https://github.com/enesbayraktar61/named-entity-recognition-nlp.git

cd named-entity-recognition-nlp
pip install -r requirements.txt
streamlit run app.py

---
