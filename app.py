from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import pickle


# Inisialisasi FastAPI
app = FastAPI()


# Path model dan tokenizer yang benar
model_path = "models/bi-lstm.h5"  
tokenizer_path = "models/tokenizer.pkl"  


# Memuat model dan tokenizer
model = load_model(model_path)  # Memuat model Keras (.h5)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)  # Memuat tokenizer


# Mendefinisikan input schema untuk prediksi
class PredictionInput(BaseModel):
    text: str  # Input berupa teks yang akan diprediksi


@app.post("/predict/")
async def predict(input_data: PredictionInput):
    try:
        # Tokenisasi dan padding input teks
        text = input_data.text
        sequence = tokenizer.texts_to_sequences([text])  # Tokenisasi teks
        padded_sequence = np.pad(sequence, ((0, 0), (0, 100 - len(sequence[0]))), 'constant')  # Padding agar panjangnya 100


        # Prediksi menggunakan model
        predictions = model.predict(padded_sequence)
       
        # Menentukan kelas numerik dan label
        predicted_class = np.argmax(predictions, axis=1)[0]  # Kelas numerik (misalnya, 0 atau 1)
        predicted_label = ["Ekonomi", "Hiburan", "Hukumdankriminal", "Kesehatan", "Politik", "Teknologi"][predicted_class]  # Label sesuai dengan indeks kelas
        
         # Membuat list probabilitas dan label masing-masing
        probability_details = [
            {"label": label, "probability": round(prob, 2)}
            for label, prob in zip(["Ekonomi", "Hiburan", "Hukumdankriminal", "Kesehatan", "Politik", "Teknologi"], predictions[0])
        ]

        # Mengambil probabilitas untuk kelas yang diprediksi dan membulatkan ke dua angka di belakang koma
        predicted_probability = round(predictions[0][predicted_class], 2)


        # Mengembalikan output sesuai format yang diminta
        return {
            "prediction": predicted_class,  # Prediksi kelas numerik
            "label": predicted_label,  # Prediksi label deskriptif
            "probability": probability_details  # Probabilitas untuk kelas yang diprediksi
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Model API is running!"}
