import json

import requests
import streamlit as st


def predict(text_input) -> str:
    out = requests.post(
        "http://app:3000/predict",
        headers={
            "Content-Type": "application/json",
            "accept": "application/json",
        },
        data=json.dumps({"text": text_input}),
    )
    return out.json()["toxic"]


text_input = st.text_input("Write your comment")

if st.button(label="Predict"):
    st.write(f"Your comment is toxic with a {predict(text_input):.3f} chance")
