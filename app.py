import streamlit as st
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

st.set_page_config(page_title="Error Severity Classifier", page_icon="🧠")
st.title("🧠 Error Severity Classifier (ONNX)")
st.markdown("Paste logs or email text to detect the **error severity** using a fine-tuned ONNX model 💡")

@st.cache_resource
def load_model():
    session = ort.InferenceSession("onnx_model_watchdog_ai/model.onnx")
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    return session, tokenizer

session, tokenizer = load_model()

user_input = st.text_area("📨 Paste a log or message to classify:")

if st.button("🔍 Predict"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="np", padding="max_length", truncation=True, max_length=128)
        outputs = session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        pred = np.argmax(outputs[0], axis=1)[0]
        st.success(f"🧠 Predicted Severity Class: `{pred}`")
    else:
        st.warning("⚠️ Please enter a valid message.")
