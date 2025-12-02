# -*- coding: utf-8 -*-
"""
Server minimale per predizioni Yu-Gi-Oh card classification
Deploy su Render - tutto in un file
"""

from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import os
from datetime import datetime
from pymongo import MongoClient

# Silenzia warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Variabili globali per modello e mappatura classi
MODEL = None
CLASS_MAPPING = {i: f"mock_card_{i}" for i in range(10)}  # Default mock

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
db_client = MongoClient(MONGO_URI)
db = db_client["yugioh_bot"]
scans_collection = db["scans"]

def load_model_and_mapping():
    """Carica modello e class_indices da Hugging Face"""
    global MODEL, CLASS_MAPPING
    
    try:
        import json
        
        # Download da Hugging Face (force_download per evitare cache)
        repo_id = "SeraphinFosfato/yugioh-card-type-classifier"
        print(f"📥 Download modello da Hugging Face: {repo_id}")
        
        model_path = hf_hub_download(repo_id=repo_id, filename="trained_model.h5", force_download=True)
        indices_path = hf_hub_download(repo_id=repo_id, filename="class_indices.json", force_download=True)
        
        print(f"📁 Model path: {model_path}")
        print(f"📁 Indices path: {indices_path}")
        
        MODEL = keras.models.load_model(model_path, compile=False)
        MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Dummy prediction per inizializzare metriche
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = MODEL.predict(dummy_input, verbose=0)
        
        with open(indices_path, "r") as f:
            label_to_int = json.load(f)
        CLASS_MAPPING = {v: k for k, v in label_to_int.items()}
        
        print(f"✅ Modello caricato da Hugging Face")
        print(f"✅ Input shape atteso: (224, 224, 3)")
        print(f"✅ Numero classi: {len(CLASS_MAPPING)}")
        print(f"✅ Classi: {list(CLASS_MAPPING.values())}")
        
    except Exception as e:
        print(f"⚠️ Errore caricamento modello da Hugging Face: {e}")
        print("⚠️ Modalità MOCK attiva")
        CLASS_MAPPING = {0: "Monster", 1: "Spell", 2: "Trap"}

# Carica modello all'import (funziona con gunicorn)
load_model_and_mapping()


def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocessa immagine da bytes -> array normalizzato per classificazione tipo carta
    target_size è (width, height) ma il modello vuole (height, width, channels)
    """
    try:
        # Decodifica bytes -> numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize: cv2.resize vuole (width, height)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        print(f"[DEBUG] Image shape dopo resize: {img.shape}")
        
        # Normalizza 0-1 float32
        img = img.astype(np.float32) / 255.0
        
        # Aggiungi batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        raise ValueError(f"Errore preprocessing: {str(e)}")


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "service": "yugioh-card-type-classifier",
        "model_loaded": MODEL is not None,
        "mode": "real" if MODEL is not None else "mock",
        "classes": list(CLASS_MAPPING.values()) if CLASS_MAPPING else [],
        "num_classes": len(CLASS_MAPPING)
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principale per predizioni
    
    Input JSON:
    {
        "image": "base64_encoded_image_string"
    }
    
    Output JSON:
    {
        "prediction": "Monster/Spell/Trap",
        "confidence": 0.95,
        "top_3": [
            {"label": "Monster", "confidence": 0.95},
            {"label": "Spell", "confidence": 0.03},
            {"label": "Trap", "confidence": 0.02}
        ]
    }
    """
    try:
        # Estrai immagine dal JSON
        data = request.get_json()
        
        if not data or "image" not in data:
            return jsonify({
                "error": "Campo 'image' mancante nel JSON"
            }), 400
        
        # Decodifica base64
        image_b64 = data["image"]
        # Rimuovi eventuali prefissi data URI
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        image_bytes = base64.b64decode(image_b64)
        
        # Preprocessa immagine
        img_array = preprocess_image(image_bytes)
        
        # MOCK MODE: se modello non caricato, restituisci predizione fake
        if MODEL is None:
            mock_predictions = np.random.dirichlet(np.ones(10))
            top_idx = int(np.argmax(mock_predictions))
            top_confidence = float(mock_predictions[top_idx])
            top_label = CLASS_MAPPING.get(top_idx, f"mock_card_{top_idx}")
            
            top_3_indices = np.argsort(mock_predictions)[-3:][::-1]
            top_3 = [
                {
                    "label": CLASS_MAPPING.get(int(idx), f"mock_card_{idx}"),
                    "confidence": float(mock_predictions[idx])
                }
                for idx in top_3_indices
            ]
            
            return jsonify({
                "prediction": top_label,
                "confidence": top_confidence,
                "top_3": top_3,
                "mock": True
            })
        
        # Predizione reale
        predictions = MODEL.predict(img_array, verbose=0)[0]
        
        # Top-1 prediction
        top_idx = int(np.argmax(predictions))
        top_confidence = float(predictions[top_idx])
        top_label = CLASS_MAPPING.get(top_idx, f"class_{top_idx}")
        
        # Top-3 predictions (solo 3 tipi: Monster/Spell/Trap)
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = [
            {
                "label": CLASS_MAPPING.get(int(idx), f"class_{idx}"),
                "confidence": float(predictions[idx])
            }
            for idx in top_3_indices
        ]
        
        result = {
            "prediction": top_label,
            "confidence": top_confidence,
            "top_3": top_3
        }
        
        # Salva nel database
        user_id = data.get("user_id")
        if user_id:
            scans_collection.insert_one({
                "user_id": user_id,
                "timestamp": datetime.utcnow(),
                "prediction": top_label,
                "confidence": top_confidence
            })
        
        return jsonify(result)
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "prediction_failed"
        }), 500


@app.route("/history/<user_id>", methods=["GET"])
def get_history(user_id):
    """Recupera lo storico scansioni di un utente"""
    try:
        print(f"[DEBUG] Richiesta history per user_id: {user_id}")
        print(f"[DEBUG] MongoDB URI configurato: {MONGO_URI[:20]}...")
        
        scans = list(scans_collection.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(20))
        
        print(f"[DEBUG] Trovate {len(scans)} scansioni")
        
        # Converti datetime in string
        for scan in scans:
            scan["timestamp"] = scan["timestamp"].isoformat()
        
        return jsonify({"scans": scans, "count": len(scans)})
    except Exception as e:
        print(f"[ERROR] Errore get_history: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Avvia server (modello già caricato all'import)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
