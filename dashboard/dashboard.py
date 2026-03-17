import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("AI Content Moderation System")

st.write("Detect toxic comments using AI")

comment = st.text_area("Enter a comment")

if st.button("Detect Comment"):

    if comment.strip() == "":
        st.warning("Please Enter a comment")

    else:

        response = requests.post(
            API_URL,
            json={"text": comment}
        )

        result = response.json()

        st.subheader("Toxicity Prediction")

        for label, score in result.items():
            st.write(f"{label}: {score:.2f}")
            st.progress(score)