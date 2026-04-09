import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from groq import Groq

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Trauma Support AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    AI-powered emotional distress detection system.
    
    Features:
    - Trauma Detection
    - Severity Analysis
    - AI Support
    - Wellness Guidance
    """
)

# ------------------------------
# Dataset
# ------------------------------
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

# ------------------------------
# ML Model
# ------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# ------------------------------
# Main UI
# ------------------------------
st.title("🧠 AI Trauma Detection & Wellness Support")
st.markdown("### Early emotional support using AI + LLM")

user_input = st.text_area(
    "💬 Share how you're feeling today:",
    height=150
)

if st.button("🔍 Analyze My Feelings"):

    # Prediction
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)
    confidence = model.predict_proba(user_vector).max() * 100

    if prediction[0] == 1:
        result = "⚠️ Emotional Distress Detected"
    else:
        result = "✅ Emotional State Appears Stable"

    # Severity
    text = user_input.lower()

    high_words = ["hopeless", "depressed", "suicide", "empty"]
    medium_words = ["anxious", "scared", "stressed", "lonely"]

    if any(word in text for word in high_words):
        severity = "🔴 HIGH RISK"
        wellness_score = 25
    elif any(word in text for word in medium_words):
        severity = "🟠 MEDIUM RISK"
        wellness_score = 55
    else:
        severity = "🟢 LOW RISK"
        wellness_score = 85

    # Layout columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", result)

    with col2:
        st.metric("Severity", severity)

    with col3:
        st.metric("Confidence", f"{confidence:.2f}%")

    # Wellness Meter
    st.subheader("🌿 Wellness Meter")
    st.progress(wellness_score)

    # Groq LLM
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
    A user said: "{user_input}"
    Severity: {severity}

    Give:
    1. empathetic emotional support
    2. calming advice
    3. coping suggestions
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("🤖 AI Emotional Support")
    st.write(response.choices[0].message.content)

    # Recommendations
    st.subheader("💡 Wellness Suggestions")

    if "HIGH" in severity:
        st.error("🚨 Please seek professional mental health support immediately.")
        st.write("• Contact a counselor")
        st.write("• Reach out to trusted family/friends")
        st.write("• Avoid isolation")

    elif "MEDIUM" in severity:
        st.warning("🌼 Consider self-care and emotional support.")
        st.write("• Practice breathing exercises")
        st.write("• Talk to someone you trust")
        st.write("• Rest properly")

    else:
        st.success("🌟 Keep maintaining healthy habits.")
        st.write("• Stay socially connected")
        st.write("• Continue positive routines")
        st.write("• Practice gratitude")
