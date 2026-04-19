import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 20
CONFIDENCE_THRESHOLD = 0.30

# ✅ FIX 1: encoding
with open("intents.json", "r", encoding="utf-8-sig") as file:
    data = json.load(file)

model = load_model("chat_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

def get_response(tag: str) -> str:
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I can't understand you."

while True:
    input_text = input("Enter your command -> ").strip()
    if not input_text:
        continue
    if input_text.lower() in ["exit", "quit", "bye"]:
        print("Bye!")
        break

    seq = tokenizer.texts_to_sequences([input_text.lower()])
    padded = pad_sequences(seq, maxlen=MAXLEN, truncating="post")
    probs = model.predict(padded, verbose=0)[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    # ✅ FIX 2: tag ko string banao
    tag = label_encoder.inverse_transform([idx])[0]

    # ✅ FIX 3: confidence threshold (warna random wrong answers)
    if confidence < CONFIDENCE_THRESHOLD:
        print(get_response("noanswer"))
    else:
        print(get_response(tag))
