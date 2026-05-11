from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

import os
import requests
import uuid
import time
import logging
import json
from typing import List, Optional
from pathlib import Path

# ─────────────────────────────
# LOAD ENV
# ─────────────────────────────
load_dotenv()

# ─────────────────────────────
# APP
# ─────────────────────────────
app = FastAPI(title="Rehman AI Backend", version="3.1")

# ─────────────────────────────
# CORS (IMPORTANT FIX)
# ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# LOGGING
# ─────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai")

# ─────────────────────────────
# ENV
# ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

MAX_HISTORY = 40
SESSION_TTL = 7200

# ─────────────────────────────
# MEMORY
# ─────────────────────────────
sessions = {}

HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

# ─────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────
SYSTEM_PROMPT = """
You are Rehman AI, a smart assistant.
Be helpful, clear, and use markdown formatting.
"""

# ─────────────────────────────
# FILE READER
# ─────────────────────────────
def extract_text(filename: str, content: bytes):
    ext = filename.split(".")[-1].lower()

    try:
        if ext in ["txt", "py", "js", "html", "css"]:
            return content.decode("utf-8", errors="ignore")[:8000]

        if ext == "json":
            return json.dumps(json.loads(content.decode()), indent=2)[:8000]

        return f"[File uploaded: {filename}]"

    except:
        return f"[Unreadable file: {filename}]"

# ─────────────────────────────
# SESSION HELPERS
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
# SAVE SESSION
# ─────────────────────────────
def save_session(sid: str, data: dict):
    path = HISTORY_DIR / f"{sid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────
# GROQ CALL
# ─────────────────────────────
def call_groq(messages):
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

# ─────────────────────────────
# ROUTES
# ─────────────────────────────
@app.get("/")
def home():
    return {"status": "Rehman AI running"}

@app.get("/health")
def health():
    return {"ok": True}

# ✅ FIX: history list
@app.get("/history")
def history():
    out = []
    for sid, s in sessions.items():
        out.append({
            "session_id": sid,
            "title": s["title"],
            "last_used": s["last_used"],
            "message_count": len(s["history"])
        })
    return {"sessions": out}

# ✅ FIX: session load (MISSING BEFORE)
@app.get("/session/{sid}")
def load_session(sid: str):
    return get_session(sid)

# ✅ FIX: delete session
@app.delete("/session/{sid}")
def delete_session(sid: str):
    sessions.pop(sid, None)
    file = HISTORY_DIR / f"{sid}.json"
    if file.exists():
        file.unlink()
    return {"deleted": True}

# ─────────────────────────────
# CHAT
# ─────────────────────────────
@app.post("/chat")
async def chat(
    message: str = Form(...),
    mode: str = Form("chat"),
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

    save_session(sid, session)

    return {
        "reply": reply,
        "session_id": sid,
        "session_title": session["title"]
    }