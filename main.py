from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pickle

app = FastAPI()

# Load Model dan Komponen Pendukung
model = load_model(r"C:\Users\Shandra Manuaba\Documents\SEMESTER 7\DEEP LEARNING (A)\project\models\model_klasifikasi_berita.h5")  # Sesuaikan path model Anda
with open(r"C:\Users\Shandra Manuaba\Documents\SEMESTER 7\DEEP LEARNING (A)\project\models\tokenizer.pkl", "rb") as f:  # Tokenizer yang sama saat training
    tokenizer = pickle.load(f)
with open(r"C:\Users\Shandra Manuaba\Documents\SEMESTER 7\DEEP LEARNING (A)\project\models\label_encoder.pkl", "rb") as f:  # LabelEncoder yang sama saat training
    label_encoder = pickle.load(f)

max_len = 100  # Panjang maksimum sequence saat training

# Fungsi Clean Text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Menghapus URL
    text = re.sub(r"\W+", " ", text)    # Menghapus karakter khusus
    text = re.sub(r"\d+", "", text)     # Menghapus angka
    text = text.lower().strip()         # Mengubah teks menjadi huruf kecil
    return text

# Schema Request
class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text

    # Preprocessing teks
    cleaned_text = clean_text(text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    input_data = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Prediksi dengan model
    probabilities = model.predict(input_data)
    predicted_class = np.argmax(probabilities)

    # Dekode label dengan LabelEncoder
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "prediction": int(predicted_class),
        "label": predicted_label,
        "probabilities": probabilities.tolist()
    }
