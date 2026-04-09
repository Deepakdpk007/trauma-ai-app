import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from groq import Groq

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Trauma Support AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# CUSTOM CSS (Premium UI)
# ------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.block-container {
    padding-top: 2rem;
}
.big-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: white;
}
.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-bottom: 2rem;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("🧠 Trauma Support AI")
st.sidebar.info("""
Advanced emotional wellness assistant powered by:
- Machine Learning
- Groq LLM
- Knowledge Base
- Wellness Guidance
""")

# ------------------------------
# KNOWLEDGE BASE
# ------------------------------
with open("knowledge_base.txt", "r") as file:
    knowledge_base = file.read()

# ------------------------------
# DATASET
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
        "Everything is fine",
        "I feel lonely and broken",
        "Nobody understands me",
        "I feel worthless",
        "I want to disappear",
        "Today was amazing",
        "I love spending time with friends",
        "Feeling peaceful and calm",
        "I feel emotionally exhausted",
        "I cry every night",
        "I feel trapped in my thoughts"
    ],
    'label': [
        1,0,1,0,1,0,1,0,
        1,1,1,1,0,0,0,1,1,1
    ]
}

df = pd.DataFrame(data)

# ------------------------------
# ML MODEL
# ------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# ------------------------------
# MAIN HEADER
# ------------------------------
st.markdown('<div class="big-title">🧠 Trauma Detection & Wellness AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced emotional support powered by AI + LLM</div>', unsafe_allow_html=True)

# ------------------------------
# INPUT SECTION
# ------------------------------
user_input = st.text_area(
    "💬 Share how you're feeling today:",
    height=180
)

# ------------------------------
# ANALYZE BUTTON
# ------------------------------
if st.button("🔍 Analyze My Feelings"):

    # Prediction
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)
    confidence = model.predict_proba(user_vector).max() * 100

    if prediction[0] == 1:
        result = "⚠️ Emotional Distress Detected"
    else:
        result = "✅ Emotional State Appears Stable"

    # Severity Logic
    text = user_input.lower()

    high_words = [
        "hopeless", "depressed", "suicide", "empty",
        "worthless", "disappear", "cry"
    ]

    medium_words = [
        "anxious", "scared", "stressed",
        "lonely", "exhausted", "trapped"
    ]

    if any(word in text for word in high_words):
        severity = "🔴 HIGH RISK"
        wellness_score = 25
    elif any(word in text for word in medium_words):
        severity = "🟠 MEDIUM RISK"
        wellness_score = 55
    else:
        severity = "🟢 LOW RISK"
        wellness_score = 85

    # Metrics
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

    # Groq API
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""
    User said: "{user_input}"
    Severity: {severity}

    Knowledge base:
    {knowledge_base}

    Give:
    1. empathetic emotional support
    2. calming advice
    3. coping suggestions
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    # AI Response Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Emotional Support")
    st.write(response.choices[0].message.content)
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)
