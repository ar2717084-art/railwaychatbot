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
import io

from typing import Optional, List
from pathlib import Path

# ─────────────────────────────────────────────
# Load ENV
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("nova")

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Nova AI Backend",
    version="3.0"
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# ENV Variables
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY not found")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "llama-3.3-70b-versatile"

MAX_HISTORY = 40
SESSION_TTL = 7200

# ─────────────────────────────────────────────
# Session Store
# ─────────────────────────────────────────────
sessions = {}

# ─────────────────────────────────────────────
# Chat History Storage
# ─────────────────────────────────────────────
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Save Session
# ─────────────────────────────────────────────
def save_session_to_disk(session_id: str, session: dict):
    try:
        path = HISTORY_DIR / f"{session_id}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

    except Exception as e:
        log.warning(f"Save session failed: {e}")

# ─────────────────────────────────────────────
# Load Session
# ─────────────────────────────────────────────
def load_session_from_disk(session_id: str):

    try:
        path = HISTORY_DIR / f"{session_id}.json"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    except Exception as e:
        log.warning(f"Load session failed: {e}")

    return None

# ─────────────────────────────────────────────
# List Sessions
# ─────────────────────────────────────────────
def list_all_sessions():

    results = []

    for path in sorted(
        HISTORY_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ):

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            results.append({
                "session_id": path.stem,
                "title": data.get("title", "New Conversation"),
                "created": data.get("created", 0),
                "last_used": data.get("last_used", 0),
                "message_count": len(data.get("history", [])),
            })

        except:
            pass

    return results[:50]

# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Nova, a smart AI assistant.

Be helpful, accurate, and professional.
Use markdown formatting.
"""

# ─────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────
MODE_INSTRUCTIONS = {
    "chat": lambda msg: msg,
    "info": lambda msg: f"Explain clearly:\n\n{msg}",
    "translate": lambda msg: f"Translate:\n\n{msg}",
    "code": lambda msg: f"Provide production-ready code:\n\n{msg}",
    "summarize": lambda msg: f"Summarize:\n\n{msg}",
}

# ─────────────────────────────────────────────
# File Reader
# ─────────────────────────────────────────────
def extract_text_from_file(filename: str, content: bytes):

    ext = filename.lower().split(".")[-1]

    try:

        if ext == "txt":
            return content.decode("utf-8", errors="replace")

        elif ext == "json":
            text = content.decode("utf-8", errors="replace")

            try:
                data = json.loads(text)

                return json.dumps(
                    data,
                    indent=2,
                    ensure_ascii=False
                )[:8000]

            except:
                return text[:8000]

        elif ext == "py":
            return content.decode("utf-8", errors="replace")

        elif ext in ["js", "ts", "html", "css"]:
            return content.decode("utf-8", errors="replace")[:8000]

        else:
            return f"[Uploaded file: {filename}]"

    except Exception as e:
        return f"[Could not read file: {str(e)}]"

# ─────────────────────────────────────────────
# Session Cleanup
# ─────────────────────────────────────────────
def clean_sessions():

    now = time.time()

    expired = [
        sid for sid, data in sessions.items()
        if now - data["last_used"] > SESSION_TTL
    ]

    for sid in expired:
        del sessions[sid]

# ─────────────────────────────────────────────
# Get Session
# ─────────────────────────────────────────────
def get_session(session_id: str):

    clean_sessions()

    if session_id not in sessions:

        saved = load_session_from_disk(session_id)

        if saved:
            sessions[session_id] = saved

        else:
            sessions[session_id] = {
                "history": [],
                "created": time.time(),
                "last_used": time.time(),
                "title": "New Conversation"
            }

    sessions[session_id]["last_used"] = time.time()

    return sessions[session_id]

# ─────────────────────────────────────────────
# Generate Title
# ─────────────────────────────────────────────
def generate_title(message: str):

    words = message.strip().split()

    title = " ".join(words[:6])

    if len(words) > 6:
        title += "..."

    return title[:50]

# ─────────────────────────────────────────────
# Call GROQ
# ─────────────────────────────────────────────
def call_groq(messages, temperature=0.7):

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        },
        timeout=45
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    data = response.json()

    return data["choices"][0]["message"]["content"]

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def home():

    return {
        "status": "Nova AI Running 🚀",
        "model": MODEL,
        "sessions": len(sessions)
    }

@app.get("/health")
def health():

    return {"ok": True}

@app.get("/history")
def history():

    return {
        "sessions": list_all_sessions()
    }

@app.post("/chat")
async def chat(
    message: str = Form(...),
    mode: str = Form("chat"),
    session_id: str = Form(None),
    files: List[UploadFile] = File(default=[])
):

    try:

        sid = session_id or str(uuid.uuid4())

        session = get_session(sid)

        history = session["history"]

        mode = mode if mode in MODE_INSTRUCTIONS else "chat"

        file_contents = []

        for upload in files:

            raw = await upload.read()

            extracted = extract_text_from_file(
                upload.filename,
                raw
            )

            file_contents.append(
                f"=== {upload.filename} ===\n{extracted}"
            )

        build = MODE_INSTRUCTIONS[mode]

        full_input = build(message)

        if file_contents:
            full_input += "\n\n" + "\n\n".join(file_contents)

        if not history:
            session["title"] = generate_title(message)

        history.append({
            "role": "user",
            "content": full_input
        })

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            *history
        ]

        reply = call_groq(messages)

        history.append({
            "role": "assistant",
            "content": reply
        })

        session["history"] = history[-MAX_HISTORY:]

        save_session_to_disk(sid, session)

        return {
            "reply": reply,
            "session_id": sid,
            "title": session["title"]
        }

    except Exception as e:

        log.exception(str(e))

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# ─────────────────────────────────────────────
# Railway Startup
# ─────────────────────────────────────────────
if __name__ == "__main__":

    import uvicorn

    PORT = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT
    )