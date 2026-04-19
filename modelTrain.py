import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# Load JSON
with open("intents.json", "r", encoding="utf-8-sig") as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []

# Prepare data
for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern.lower())
        training_labels.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Tokenizer
vocab_size = 1000
max_len = 20
embedding_dim = 16
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post')

# Model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(len(labels), activation="softmax")
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=["accuracy"]
)

# Early Stopping
early_stop = EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True
)

# Train
model.fit(
    padded_sequences,
    np.array(training_labels),
    epochs=300,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save("chat_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Training Complete ✅")
