import streamlit as st
import json
import pickle
import numpy as np
import pyttsx3
import speech_recognition as sr

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= LOAD FILES =================

with open("intents.json") as file:
    data = json.load(file)

model = load_model("chat_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as enc:
    label_encoder = pickle.load(enc)

# ================= SPEAK FUNCTION =================

def speak(text):
    engine = pyttsx3.init()

    rate = engine.getProperty('rate')
    engine.setProperty('rate', 150)   # 🔥 Change speed here

    engine.say(text)
    engine.runAndWait()

# ================= VOICE INPUT =================

def listen():

    r = sr.Recognizer()

    with sr.Microphone() as source:

        st.info("🎤 Listening... Speak now")

        # Adjust noise
        r.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = r.listen(
                source,
                timeout=3,             # wait max 3 sec for speech start
                phrase_time_limit=5    # max speaking time 5 sec
            )

        except sr.WaitTimeoutError:
            return ""

    try:
        text = r.recognize_google(audio, language="en-in")
        return text.lower()

    except:
        return ""

# ================= MODEL FUNCTION =================

def get_response(user_text):

    padded = pad_sequences(
        tokenizer.texts_to_sequences([user_text]),
        maxlen=20,
        truncating="post"
    )

    result = model.predict(padded, verbose=0)

    confidence = np.max(result)
    tag = label_encoder.inverse_transform([np.argmax(result)])[0]

    if confidence < 0.3:
        return "Sorry I did not understand"

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])

    return "No response found"

# ================= UI =================

st.set_page_config(page_title="AUDIO Q/A For Students ", page_icon="🤖")

st.title("🤖AUDIO Q/A For Students")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ================= BUTTONS =================

col1, col2 = st.columns(2)

# 🎤 MIC BUTTON
if col1.button("🎤 Speak"):
    voice_text = listen()

    if voice_text != "":
        st.session_state.messages.append({"role": "user", "content": voice_text})

        with st.chat_message("user"):
            st.write(voice_text)

        response = get_response(voice_text)

        speak(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.write(response)

# ⌨ TEXT INPUT
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)

    speak(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)











