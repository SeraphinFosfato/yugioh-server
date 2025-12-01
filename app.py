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

# Silenzia warning TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

app = Flask(__name__)

# Variabili globali per modello e mappatura classi
MODEL = None
CLASS_MAPPING = {i: f"mock_card_{i}" for i in range(10)}  # Default mock

def load_model_and_mapping():
    """Carica modello e class_indices al primo avvio"""
    global MODEL, CLASS_MAPPING
    
    model_path = os.getenv("MODEL_PATH", "models/trained_model.h5")
    indices_path = "models/class_indices.json"
    
    # Prova a caricare il modello
    if os.path.exists(model_path) and os.path.exists(indices_path):
        try:
            import json
            MODEL = keras.models.load_model(model_path)
            with open(indices_path, "r") as f:
                label_to_int = json.load(f)
            CLASS_MAPPING = {v: k for k, v in label_to_int.items()}
            print(f"✅ Modello caricato da {model_path}")
            return
        except Exception as e:
            print(f"⚠️ Errore caricamento modello: {e}")
    
    # Modalità MOCK per testing
    print("⚠️ Modello non trovato - Modalità MOCK attiva")

# Carica modello all'import (funziona con gunicorn)
load_model_and_mapping()


def preprocess_image(image_bytes, target_size=(813, 1185)):
    """
    Preprocessa immagine da bytes -> array normalizzato
    Stesso preprocessing di data_preprocessing.py
    """
    try:
        # Decodifica bytes -> numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize alla dimensione del modello
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
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
        "service": "yugioh-card-classifier",
        "model_loaded": MODEL is not None
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
        "prediction": "card_id",
        "confidence": 0.95,
        "top_5": [
            {"label": "card_id_1", "confidence": 0.95},
            {"label": "card_id_2", "confidence": 0.03},
            ...
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
            
            top_5_indices = np.argsort(mock_predictions)[-5:][::-1]
            top_5 = [
                {
                    "label": CLASS_MAPPING.get(int(idx), f"mock_card_{idx}"),
                    "confidence": float(mock_predictions[idx])
                }
                for idx in top_5_indices
            ]
            
            return jsonify({
                "prediction": top_label,
                "confidence": top_confidence,
                "top_5": top_5,
                "mock": True
            })
        
        # Predizione reale
        predictions = MODEL.predict(img_array, verbose=0)[0]
        
        # Top-1 prediction
        top_idx = int(np.argmax(predictions))
        top_confidence = float(predictions[top_idx])
        top_label = CLASS_MAPPING.get(top_idx, f"class_{top_idx}")
        
        # Top-5 predictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        top_5 = [
            {
                "label": CLASS_MAPPING.get(int(idx), f"class_{idx}"),
                "confidence": float(predictions[idx])
            }
            for idx in top_5_indices
        ]
        
        return jsonify({
            "prediction": top_label,
            "confidence": top_confidence,
            "top_5": top_5
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "prediction_failed"
        }), 500


if __name__ == "__main__":
    # Avvia server (modello già caricato all'import)
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
