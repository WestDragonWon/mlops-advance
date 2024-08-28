# - pip install streamlit

import streamlit as st # MVP 만들기 좋은 라이브러리

st.title('TinyBERT Model Test Web')
st.header("Check your sentiment")

text = st.text_area("Input your sentence", "게시글 내용")

clicked = st.button("확인")

import torch
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = pipeline('text-classification',  model='ml-models/tinybert-sentiment-analysis', device=device)

if clicked:
    with st.spinner("분석중입니다."):
        res = classifier(text)
        st.write(res)

# python -m streamlit run front-streamlit.py