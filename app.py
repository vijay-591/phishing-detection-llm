import streamlit as st
from src.predict import predict_email

st.set_page_config(page_title="Phishing Detection")

st.title("🔐 AI-Based Phishing Detection Using LLMs")

email = st.text_area("Enter Email / Message")

if st.button("Detect"):
    if email.strip() == "":
        st.warning("Please enter text")
    else:
        st.success(predict_email(email))
