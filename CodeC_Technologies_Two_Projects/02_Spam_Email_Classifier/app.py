from pathlib import Path
import joblib
import streamlit as st

MODEL_FILE = Path(__file__).parent / "model" / "spam_classifier.joblib"

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Classifies a message as spam or legitimate using TF-IDF + Multinomial Naive Bayes.")

if not MODEL_FILE.exists():
    st.warning("The trained model is not available yet.")
    st.code("python train_model.py", language="bash")
    st.stop()

model = joblib.load(MODEL_FILE)

message = st.text_area(
    "Enter an email/SMS message",
    height=180,
    placeholder="Example: Congratulations! You won a free prize..."
)

if st.button("Classify Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        label = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities)

        if label == "spam":
            st.error(f"🚨 SPAM — confidence: {confidence:.2%}")
        else:
            st.success(f"✅ NOT SPAM — confidence: {confidence:.2%}")
