# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from twilio.twiml.messaging_response import MessagingResponse
# from pydantic import BaseModel
# from dotenv import load_dotenv
# import requests, uuid, json, os
# from pathlib import Path
# from datetime import datetime

# load_dotenv()
# app = FastAPI(title="Digital Arrest Shield API")

# app.add_middleware(CORSMiddleware,
#     allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# MODEL_URL = os.getenv("MODEL_API_URL", "")
# Path("reports").mkdir(exist_ok=True)

# class TextInput(BaseModel):
#     text: str

# class ReportInput(BaseModel):
#     transcript: str
#     device_id: str = "unknown"
#     timestamp: str = ""

# @app.get("/")
# def root():
#     return {"status": "Digital Arrest Shield API running", "version": "1.0"}

# @app.post("/classify")
# def classify(input: TextInput):
#     if not input.text:
#         return {"error": "no text provided"}
#     if MODEL_URL:
#         try:
#             res = requests.post(MODEL_URL, json={"text": input.text}, timeout=5)
#             return res.json()
#         except Exception:
#             pass
#     return mock_classify(input.text)

# def mock_classify(text):
#     scam_words = [
#         "digital arrest", "transfer money", "cbi officer",
#         "ncb", "warrant", "safe account", "do not tell anyone",
#         "cyber crime", "money laundering", "remain on call"
#     ]
#     text_lower = text.lower()
#     triggered = [w for w in scam_words if w in text_lower]
#     hit = len(triggered) > 0
#     return {
#         "score": 0.95 if hit else 0.08,
#         "label": "scam" if hit else "safe",
#         "risk": "HIGH" if hit else "LOW",
#         "triggered_phrases": triggered,
#         "advice": "Hang up immediately. Call 1930." if hit else "Appears safe."
#     }

# @app.post("/bot/webhook")
# async def bot_webhook(request: Request):
#     form = await request.form()
#     user_msg = form.get("Body", "")
#     result = mock_classify(user_msg)
#     resp = MessagingResponse()
#     if result["risk"] == "HIGH":
#         reply = (
#             "ALERT - SCAM DETECTED\n\n"
#             f"Risk Level: {result['risk']}\n"
#             f"Matched: {', '.join(result['triggered_phrases'])}\n\n"
#             "DO NOT transfer money.\n"
#             "Hang up immediately.\n"
#             "Call 1930 (Cyber Crime Helpline)\n\n"
#             "No genuine authority in India arrests via phone or video call."
#         )
#     else:
#         reply = (
#             "Message appears SAFE.\n\n"
#             "Stay alert - no genuine authority in India "
#             "conducts arrests via phone or video call.\n\n"
#             "If unsure, call 1930."
#         )
#     resp.message(reply)
#     return str(resp)

# @app.post("/report")
# def submit_report(input: ReportInput):
#     case_id = f"DAS-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
#     report = {
#         "case_id": case_id,
#         "timestamp": str(datetime.now()),
#         "transcript": input.transcript,
#         "device_id": input.device_id
#     }
#     with open(f"reports/{case_id}.json", "w") as f:
#         json.dump(report, f, indent=2)
#     return {
#         "case_id": case_id,
#         "status": "received",
#         "message": "Report recorded. Share this case_id with cyber authorities."
#     }

# @app.get("/report/{case_id}")
# def get_report(case_id: str):
#     path = f"reports/{case_id}.json"
#     if not os.path.exists(path):
#         return {"error": "Case not found"}
#     with open(path) as f:
#         return json.load(f)
import pickle
import warnings
import uuid
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings('ignore')

app = FastAPI(title="Digital Arrest Shield API")

# ── CORS — allows HTML frontend to call this API ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load .pkl model once at startup ──────────────────────────
print("=" * 55)
print("     DIGITAL ARREST SHIELD — Backend Starting")
print("=" * 55)
print("\n⏳ Loading ML model...")

with open('digital_arrest_model_final.pkl', 'rb') as f:
    model_data = pickle.load(f)

ML_MODEL = model_data['model']
TFIDF    = model_data['tfidf']

print(f"✅ Model loaded — Accuracy: {model_data['accuracy']}")
print(f"✅ Backend ready at http://localhost:8000")
print("=" * 55)

# ── Scam trigger phrases ──────────────────────────────────────
SCAM_PHRASES = [
    "digital arrest", "arrest warrant", "cbi officer",
    "ed officer", "ncb officer", "rbi official",
    "income tax officer", "cyber crime police",
    "transfer money", "upi transfer", "safe account",
    "pay immediately", "do not tell anyone",
    "stay on call", "you are under arrest",
    "case filed against you", "freeze your account",
    "digital arrest mein hain", "turant transfer karo",
    "parivaar ko mat batana", "jail bhej denge",
    "enforcement directorate", "cbi", "money laundering",
]

# ── In-memory report store ────────────────────────────────────
reports_store = {}

# ── Input models ─────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class ReportInput(BaseModel):
    original_text    : str
    label            : str
    risk             : str
    score            : float
    triggered_phrases: list
    advice           : str

# ── /classify endpoint ────────────────────────────────────────
@app.post("/classify")
def classify(input: TextInput):
    original_text = input.text.strip()

    # Step 1 — Try translation if needed
    try:
        from langdetect import detect
        detected_lang = detect(original_text)
    except:
        detected_lang = "en"

    try:
        if detected_lang != "en":
            from deep_translator import GoogleTranslator
            translated_text = GoogleTranslator(
                source='auto',
                target='english'
            ).translate(original_text)
        else:
            translated_text = original_text
    except:
        translated_text = original_text

    # Step 2 — Run through ML model
    clean_text = translated_text.lower()
    vec        = TFIDF.transform([clean_text])
    prediction = ML_MODEL.predict(vec)[0]
    confidence = ML_MODEL.predict_proba(vec)[0][prediction] * 100

    # Step 3 — Find triggered phrases
    triggered = [
        phrase for phrase in SCAM_PHRASES
        if phrase in clean_text
        or phrase in original_text.lower()
    ]

    # Step 4 — Risk level
    if prediction == 1 and confidence >= 70:
        risk = "HIGH"
    elif prediction == 1 and confidence >= 50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Step 5 — Advice
    if prediction == 1:
        advice = "Hang up immediately. Do not transfer any money. Call 1930."
    else:
        advice = "No threat detected. Stay alert and never share personal details."

    return {
        "original_text"     : original_text,
        "detected_language" : detected_lang,
        "translated_text"   : translated_text,
        "score"             : round(confidence / 100, 2),
        "label"             : "scam" if prediction == 1 else "normal",
        "risk"              : risk,
        "triggered_phrases" : triggered,
        "advice"            : advice
    }

# ── /report endpoint ─────────────────────────────────────────
@app.post("/report")
def report(input: ReportInput):
    case_id = f"DAS-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    reports_store[case_id] = {
        "case_id"          : case_id,
        "timestamp"        : datetime.now().isoformat(),
        "original_text"    : input.original_text,
        "label"            : input.label,
        "risk"             : input.risk,
        "score"            : input.score,
        "triggered_phrases": input.triggered_phrases,
        "advice"           : input.advice,
        "status"           : "reported"
    }

    return {
        "case_id"   : case_id,
        "message"   : "Report submitted successfully",
        "timestamp" : datetime.now().isoformat(),
        "next_steps": [
            "Case registered with Digital Arrest Shield",
            "Report to cybercrime.gov.in",
            "National Cyber Crime Helpline: 1930"
        ]
    }

# ── /reports endpoint — see all reports ──────────────────────
@app.get("/reports")
def get_reports():
    return {
        "total"  : len(reports_store),
        "reports": list(reports_store.values())
    }

# ── /health endpoint ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status" : "online",
        "model"  : "digital_arrest_shield_v3",
        "accuracy": model_data['accuracy']
    }
