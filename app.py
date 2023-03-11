import streamlit as st
# from transformers import AutoTokenizer, T5ForConditionalGeneration
from boolqgen import BoolQGen
qe = BoolQGen()

# Set the app title
st.title("True/False Question Generator")
# Add a text input for the context
context = st.text_input("Enter the context:")
# Add a button to generate the true/false question
if st.button("Generate Question"):
    payload = {
        "input_text": context
    }
    output = qe.predict_boolq(payload)
    for i in range(len(output['Boolean Questions'])):
        st.write(output['Boolean Questions'][i] + " (True/False)")
