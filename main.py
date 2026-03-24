import pickle
import warnings
import uuid
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings('ignore')

app = FastAPI(title="Digital Arrest Shield API", version="1.0")

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

try:
    with open('digital_arrest_model_final.pkl', 'rb') as f:
        model_data = pickle.load(f)

    ML_MODEL = model_data['model']
    TFIDF    = model_data['tfidf']
    print(f"✅ Model loaded — Accuracy: {model_data['accuracy']}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    ML_MODEL = None
    TFIDF    = None
    model_data = {'accuracy': 'N/A'}

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

# ── / root endpoint ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status" : "Digital Arrest Shield API running",
        "version": "1.0"
    }

# ── /health endpoint ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"  : "online" if ML_MODEL is not None else "model_missing",
        "model"   : "digital_arrest_shield_v3",
        "accuracy": model_data['accuracy'],
        "pkl_loaded": ML_MODEL is not None
    }

# ── /classify endpoint ────────────────────────────────────────
@app.post("/classify")
def classify(input: TextInput):

    if ML_MODEL is None:
        return {
            "error"  : "Model not loaded. Please check if .pkl file is present.",
            "label"  : "error",
            "risk"   : "UNKNOWN",
            "score"  : 0,
            "triggered_phrases": [],
            "advice" : "Backend model missing. Contact support."
        }

    original_text = input.text.strip()

    # Step 1 — Detect language
    try:
        from langdetect import detect
        detected_lang = detect(original_text)
    except:
        detected_lang = "en"

    # Step 2 — Translate if not English
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

    # Step 3 — Run through ML model
    clean_text = translated_text.lower()
    vec        = TFIDF.transform([clean_text])
    prediction = ML_MODEL.predict(vec)[0]
    confidence = ML_MODEL.predict_proba(vec)[0][prediction] * 100

    # Step 4 — Find triggered phrases
    triggered = [
        phrase for phrase in SCAM_PHRASES
        if phrase in clean_text
        or phrase in original_text.lower()
    ]

    # Step 5 — Risk level
    if prediction == 1 and confidence >= 70:
        risk = "HIGH"
    elif prediction == 1 and confidence >= 50:
        risk = "MEDIUM"
    elif prediction == 1:
        risk = "LOW"
    else:
        risk = "NONE"

    # Step 6 — Advice
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

# ── /report/{case_id} endpoint — get single report ───────────
@app.get("/report/{case_id}")
def get_report(case_id: str):
    if case_id not in reports_store:
        return {"error": "Case ID not found"}
    return reports_store[case_id]

# ── /reports endpoint — see all reports ──────────────────────
@app.get("/reports")
def get_reports():
    return {
        "total"  : len(reports_store),
        "reports": list(reports_store.values())
    }
