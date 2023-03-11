# Generating Boolean Questions using PyTorch and Transformers

This repository contains code for generating True/False questions from input text using PyTorch and the Transformers library. This can be useful for generating quiz questions, or for automatically checking understanding of a given text.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install torch
pip install transformers
pip install streamlit
```

## Usage
To run the Streamlit app, navigate to the directory where the code is located and run:
```bash
streamlit run app.py
```
This will launch the app in your default web browser.

You can modify the app layout and behavior by editing the code in `app.py`.

The `predict_boolq` function uses a pre-trained T5 model to generate True/False questions from input text. This function takes as input a string of text and the number of questions to generate and returns a list of generated questions

## Conclusion
This project has demonstrated how to generate True/False questions from input text using PyTorch and the Transformers library. This repository has provided a simple Streamlit app that allows users to generate Boolean questions from their own input text.

This technology has many potential applications, such as generating quiz questions, automatically checking to understand of a given text, or assisting in language learning.

## Demo
![Demo Quiz Question](https://github.com/Hirkani2003/Generate_Bool_Questions/blob/main/demo_3.PNG)
