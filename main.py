from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests, uuid, json, os
from pathlib import Path
from datetime import datetime

load_dotenv()
app = FastAPI(title="Digital Arrest Shield API")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_URL = os.getenv("MODEL_API_URL", "")
Path("reports").mkdir(exist_ok=True)

class TextInput(BaseModel):
    text: str

class ReportInput(BaseModel):
    transcript: str
    device_id: str = "unknown"
    timestamp: str = ""

@app.get("/")
def root():
    return {"status": "Digital Arrest Shield API running", "version": "1.0"}

@app.post("/classify")
def classify(input: TextInput):
    if not input.text:
        return {"error": "no text provided"}
    if MODEL_URL:
        try:
            res = requests.post(MODEL_URL, json={"text": input.text}, timeout=5)
            return res.json()
        except Exception:
            pass
    return mock_classify(input.text)

def mock_classify(text):
    scam_words = [
        "digital arrest", "transfer money", "cbi officer",
        "ncb", "warrant", "safe account", "do not tell anyone",
        "cyber crime", "money laundering", "remain on call"
    ]
    text_lower = text.lower()
    triggered = [w for w in scam_words if w in text_lower]
    hit = len(triggered) > 0
    return {
        "score": 0.95 if hit else 0.08,
        "label": "scam" if hit else "safe",
        "risk": "HIGH" if hit else "LOW",
        "triggered_phrases": triggered,
        "advice": "Hang up immediately. Call 1930." if hit else "Appears safe."
    }

@app.post("/bot/webhook")
async def bot_webhook(request: Request):
    form = await request.form()
    user_msg = form.get("Body", "")
    result = mock_classify(user_msg)
    resp = MessagingResponse()
    if result["risk"] == "HIGH":
        reply = (
            "ALERT - SCAM DETECTED\n\n"
            f"Risk Level: {result['risk']}\n"
            f"Matched: {', '.join(result['triggered_phrases'])}\n\n"
            "DO NOT transfer money.\n"
            "Hang up immediately.\n"
            "Call 1930 (Cyber Crime Helpline)\n\n"
            "No genuine authority in India arrests via phone or video call."
        )
    else:
        reply = (
            "Message appears SAFE.\n\n"
            "Stay alert - no genuine authority in India "
            "conducts arrests via phone or video call.\n\n"
            "If unsure, call 1930."
        )
    resp.message(reply)
    return str(resp)

@app.post("/report")
def submit_report(input: ReportInput):
    case_id = f"DAS-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
    report = {
        "case_id": case_id,
        "timestamp": str(datetime.now()),
        "transcript": input.transcript,
        "device_id": input.device_id
    }
    with open(f"reports/{case_id}.json", "w") as f:
        json.dump(report, f, indent=2)
    return {
        "case_id": case_id,
        "status": "received",
        "message": "Report recorded. Share this case_id with cyber authorities."
    }

@app.get("/report/{case_id}")
def get_report(case_id: str):
    path = f"reports/{case_id}.json"
    if not os.path.exists(path):
        return {"error": "Case not found"}
    with open(path) as f:
        return json.load(f)