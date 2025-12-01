# Server Yu-Gi-Oh Card Classifier

Server minimale Flask per predizioni di carte Yu-Gi-Oh.

## Setup Locale

```bash
pip install -r requirements.txt
python app.py
```

## Deploy su Render

### 1. Prepara i file
- Copia `trained_model.h5` e `class_indices.json` da `Colab_Demo/models/` a `server/models/`
- Decommenta righe 23-31 in `app.py` per caricare il modello

### 2. Deploy
1. Crea nuovo Web Service su Render
2. Collega repository GitHub
3. Root Directory: `server`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app`
6. Environment: Python 3

### 3. Configura Bot
Nel file `.env` del bot, imposta:
```
CNN_SERVER_URL=https://your-app.onrender.com
```

## API Endpoints

### GET /
Health check del server

**Response:**
```json
{
  "status": "online",
  "service": "yugioh-card-classifier",
  "model_loaded": true
}
```

### POST /predict
Predizione carta da immagine

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "prediction": "12345",
  "confidence": 0.95,
  "top_5": [
    {"label": "12345", "confidence": 0.95},
    {"label": "67890", "confidence": 0.03}
  ]
}
```

## Integrazione Bot

Il bot invia immagini in base64 a `/predict` e riceve le predizioni.
Dimensioni immagine: 813x1185 (come nel training).

## Checklist Deploy

- [ ] Copiare `models/trained_model.h5` in `server/models/`
- [ ] Copiare `models/class_indices.json` in `server/models/`
- [ ] Decommentare caricamento modello in `app.py` (righe 23-31)
- [ ] Push su GitHub
- [ ] Deploy su Render
- [ ] Aggiornare `CNN_SERVER_URL` nel bot
