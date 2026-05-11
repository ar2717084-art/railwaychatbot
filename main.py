from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
import uuid
import time
import json
from typing import List
from pathlib import Path

load_dotenv()

app = FastAPI(title="Rehman AI Backend", version="4.1")

# ─────────────────────────────
# CORS (IMPORTANT FOR VERCEL)
# ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# ENV
# ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

MAX_HISTORY = 40

sessions = {}
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
You are Rehman AI, a smart assistant.
Be helpful, clear, and use markdown formatting.
"""

# ─────────────────────────────
# FILE READER
# ─────────────────────────────
def extract_text(filename: str, content: bytes):
    try:
        if filename.endswith((".txt", ".py", ".js", ".html", ".css")):
            return content.decode("utf-8", errors="ignore")[:8000]

        if filename.endswith(".json"):
            return json.dumps(json.loads(content.decode()), indent=2)[:8000]

        return f"[File uploaded: {filename}]"
    except:
        return f"[Unreadable file: {filename}]"

# ─────────────────────────────
# SESSION
# ─────────────────────────────
def get_session(sid: str):
    if sid not in sessions:
        sessions[sid] = {
            "history": [],
            "created": time.time(),
            "last_used": time.time(),
            "title": "New Chat"
        }
    sessions[sid]["last_used"] = time.time()
    return sessions[sid]

# ─────────────────────────────
# GROQ CALL
# ─────────────────────────────
def call_groq(messages):
    try:
        res = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048
            }
        )
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# ─────────────────────────────
# ROUTES
# ─────────────────────────────
@app.get("/")
def home():
    return {"status": "Rehman AI running on Render"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/history")
def history():
    return {
        "sessions": [
            {
                "session_id": sid,
                "title": s["title"],
                "last_used": s["last_used"],
                "message_count": len(s["history"])
            }
            for sid, s in sessions.items()
        ]
    }

@app.get("/session/{sid}")
def load_session(sid: str):
    return get_session(sid)

@app.delete("/session/{sid}")
def delete_session(sid: str):
    sessions.pop(sid, None)
    return {"deleted": True}

@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(None),
    files: List[UploadFile] = File(default=[])
):

    sid = session_id or str(uuid.uuid4())
    session = get_session(sid)

    file_text = ""
    for f in files:
        raw = await f.read()
        file_text += extract_text(f.filename, raw) + "\n"

    user_input = message + "\n\n" + file_text

    session["history"].append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session["history"][-MAX_HISTORY:]

    reply = call_groq(messages)

    session["history"].append({"role": "assistant", "content": reply})
    session["title"] = message[:40]

    return {
        "reply": reply,
        "session_id": sid,
        "session_title": session["title"]
    }
    