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
from io import BytesIO

# File readers
import pdfplumber
from docx import Document

# ─────────────────────────────
# LOAD ENV
# ─────────────────────────────
load_dotenv()

# ─────────────────────────────
# APP
# ─────────────────────────────
app = FastAPI(
    title="Rehman AI Backend",
    version="5.0"
)

# ─────────────────────────────
# CORS
# ─────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with vercel domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# ENV VARIABLES
# ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# ─────────────────────────────
# GROQ CONFIG
# ─────────────────────────────
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "llama-3.3-70b-versatile"

MAX_HISTORY = 40

# ─────────────────────────────
# MEMORY STORAGE
# ─────────────────────────────
sessions = {}

# ─────────────────────────────
# CHAT HISTORY FOLDER
# ─────────────────────────────
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

# ─────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────
SYSTEM_PROMPT = """
You are Rehman AI.

You are a smart, modern, helpful AI assistant.

Rules:
- Be clear and helpful
- Use markdown formatting
- Give clean code blocks
- Explain simply
- Be conversational
- Be accurate
- Help with coding, writing, summaries, ideas, and explanations
"""

# ─────────────────────────────
# FILE TEXT EXTRACTION
# ─────────────────────────────
def extract_text(filename: str, content: bytes):

    try:

        filename_lower = filename.lower()

        # ───────────────── TXT / CODE ─────────────────
        if filename_lower.endswith((
            ".txt",
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".html",
            ".css",
            ".json",
            ".md",
            ".csv"
        )):
            return content.decode(
                "utf-8",
                errors="ignore"
            )[:12000]

        # ───────────────── PDF ─────────────────
        if filename_lower.endswith(".pdf"):

            text = ""

            with pdfplumber.open(BytesIO(content)) as pdf:

                for page in pdf.pages:

                    extracted = page.extract_text()

                    if extracted:
                        text += extracted + "\n"

            return text[:12000]

        # ───────────────── DOCX ─────────────────
        if filename_lower.endswith(".docx"):

            doc = Document(BytesIO(content))

            text = "\n".join(
                [p.text for p in doc.paragraphs]
            )

            return text[:12000]

        # ───────────────── XLSX ─────────────────
        if filename_lower.endswith((".xlsx", ".xls")):
            return f"[Excel file uploaded: {filename}]"

        # ───────────────── IMAGE ─────────────────
        if filename_lower.endswith((
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        )):
            return f"[Image uploaded: {filename}]"

        return f"[File uploaded: {filename}]"

    except Exception as e:

        return f"[Unreadable file: {filename}] Error: {str(e)}"

# ─────────────────────────────
# SESSION HANDLER
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
# GROQ API CALL
# ─────────────────────────────
def call_groq(messages):

    try:

        response = requests.post(
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
            },

            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "⚠️ Request timed out."

    except requests.exceptions.RequestException as e:
        return f"⚠️ API Error: {str(e)}"

    except Exception as e:
        return f"⚠️ Server Error: {str(e)}"

# ─────────────────────────────
# HOME
# ─────────────────────────────
@app.get("/")
def home():

    return {
        "status": "Rehman AI Backend Running",
        "model": MODEL,
        "version": "5.0"
    }

# ─────────────────────────────
# HEALTH
# ─────────────────────────────
@app.get("/health")
def health():

    return {
        "ok": True
    }

# ─────────────────────────────
# HISTORY LIST
# ─────────────────────────────
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

# ─────────────────────────────
# LOAD SESSION
# ─────────────────────────────
@app.get("/session/{sid}")
def load_session(sid: str):

    return get_session(sid)

# ─────────────────────────────
# DELETE SESSION
# ─────────────────────────────
@app.delete("/session/{sid}")
def delete_session(sid: str):

    sessions.pop(sid, None)

    return {
        "deleted": True
    }

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

    # ───────────────── SESSION ─────────────────
    sid = session_id or str(uuid.uuid4())

    session = get_session(sid)

    # ───────────────── FILE PROCESSING ─────────────────
    file_text = ""

    for f in files:

        try:

            raw = await f.read()

            extracted = extract_text(
                f.filename,
                raw
            )

            file_text += f"""

--- Attached File: {f.filename} ---

{extracted}

"""

        except Exception as e:

            file_text += f"\n[Could not read file: {f.filename}] {str(e)}\n"

    # ───────────────── MODES ─────────────────
    mode_prompts = {

        "chat":
            "Have a natural helpful conversation.",

        "info":
            "Explain clearly in simple terms.",

        "translate":
            "Translate accurately and naturally.",

        "suggest":
            "Give practical and useful suggestions.",

        "game":
            "Act like a fun game assistant.",

        "create":
            "Be highly creative and imaginative.",

        "summarize":
            "Summarize clearly and briefly.",

        "code":
            "Help with programming and coding professionally."
    }

    # ───────────────── FINAL USER INPUT ─────────────────
    final_input = f"""
Mode: {mode}

Instruction:
{mode_prompts.get(mode, "")}

User Message:
{message}

{file_text}
"""

    # ───────────────── SAVE USER MSG ─────────────────
    session["history"].append({
        "role": "user",
        "content": final_input
    })

    # ───────────────── LIMIT HISTORY ─────────────────
    trimmed_history = session["history"][-MAX_HISTORY:]

    # ───────────────── BUILD MESSAGES ─────────────────
    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

    ] + trimmed_history

    # ───────────────── GROQ RESPONSE ─────────────────
    reply = call_groq(messages)

    # ───────────────── SAVE AI RESPONSE ─────────────────
    session["history"].append({

        "role": "assistant",
        "content": reply
    })

    # ───────────────── CHAT TITLE ─────────────────
    if message.strip():

        session["title"] = message[:40]

    # ───────────────── RETURN ─────────────────
    return {

        "reply": reply,

        "session_id": sid,

        "session_title": session["title"],

        "mode": mode
    }

# ─────────────────────────────
# RUN LOCAL
# ─────────────────────────────
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    