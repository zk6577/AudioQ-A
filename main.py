import datetime
import sys
import time
import webbrowser
import pyautogui
import pyttsx3
import speech_recognition as sr
import json
import pickle
import numpy as np
import psutil
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


#  LOAD TRAINED FILES 
with open("intents.json") as file:
    data = json.load(file)

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pkl", "rb") as enc:
    label_encoder = pickle.load(enc)

model = load_model("chat_model.h5")


# = TTS =
def initialize_engine():
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 165)
    return engine


def speak(text):
    engine = initialize_engine()
    engine.say(text)
    engine.runAndWait()


# = VOICE INPUT 
def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio, language="en-in")
        print("User:", query)
    except:
        return "none"

    return query.lower()


# TIME 
def wishMe():
    hour = int(datetime.datetime.now().hour)

    if hour < 12:
        speak("Good Morning")
    elif hour < 18:
        speak("Good Afternoon")
    else:
        speak("Good Evening")


# ===== SOCIAL MEDIA =====
def social_media(cmd):
    if "facebook" in cmd:
        webbrowser.open("https://facebook.com")
    elif "instagram" in cmd:
        webbrowser.open("https://instagram.com")
    elif "whatsapp" in cmd:
        webbrowser.open("https://web.whatsapp.com")


# ===== SYSTEM STATUS =====
def condition():
    cpu = psutil.cpu_percent()
    battery = psutil.sensors_battery().percent

    speak(f"CPU usage is {cpu} percent")
    speak(f"Battery is {battery} percent")


# ===== MODEL RESPONSE =====
def model_response(query):
    padded = pad_sequences(
        tokenizer.texts_to_sequences([query]),
        maxlen=20,
        truncating="post"
    )

    result = model.predict(padded)

    confidence = np.max(result)
    tag = label_encoder.inverse_transform([np.argmax(result)])[0]

    # DEBUG PRINT
    print("Confidence:", confidence)
    print("Predicted Tag:", tag)

    # LOWER CONFIDENCE THRESHOLD
    if confidence < 0.3:
        speak("Sorry I did not understand")
        return

    for intent in data["intents"]:
        if intent["tag"] == tag:
            response = np.random.choice(intent["responses"])
            print("Bot:", response)
            speak(response)
            return

    # FALLBACK
    speak("I could not find answer in dataset")


#  MAIN LOOP 
if __name__ == "__main__":
    wishMe()

    while True:
        query = command()

        if query == "none":
            continue

        elif "facebook" in query or "instagram" in query or "whatsapp" in query:
            social_media(query)

        elif "system condition" in query:
            condition()

        elif "exit" in query:
            speak("Goodbye")
            sys.exit()

        else:
            model_response(query)
