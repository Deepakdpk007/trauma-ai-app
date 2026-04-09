import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from groq import Groq

# Dataset
data = {
    'text': [
        "I feel very sad and hopeless",
        "I am happy today",
        "I feel anxious and scared",
        "Life is going well",
        "I cannot sleep and feel stressed",
        "I am enjoying my day",
        "I feel empty and depressed",
        "Everything is fine"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# UI
st.title("🧠 AI Trauma Detection & Support System")

user_input = st.text_area("Enter your feelings:")

if st.button("Analyze"):

    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)

    if prediction[0] == 1:
        result = "⚠️ Emotional Distress Detected"
    else:
        result = "✅ Normal"

    st.subheader("Prediction")
    st.write(result)

    text = user_input.lower()

    high_words = ["hopeless", "depressed", "suicide", "empty"]
    medium_words = ["anxious", "scared", "stressed", "lonely"]

    if any(word in text for word in high_words):
        severity = "🔴 HIGH RISK"
    elif any(word in text for word in medium_words):
        severity = "🟠 MEDIUM RISK"
    else:
        severity = "🟢 LOW RISK"

    st.subheader("Severity")
    st.write(severity)

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
    A user said: "{user_input}"
    Severity: {severity}

    Give empathetic response, advice, and support.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("AI Support")
    st.write(response.choices[0].message.content)

    st.subheader("Recommendation")

    if "HIGH" in severity:
        st.error("Seek professional help immediately.")
    elif "MEDIUM" in severity:
        st.warning("Talk to someone you trust.")
    else:
        st.success("You're doing fine. Keep going.")
