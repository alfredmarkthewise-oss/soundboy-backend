"""
Soundboy AI — FastAPI Backend Server
Serves /api/mix (POST), /api/mix-data (GET), /api/chat-mix (POST)
"""

import os
import json
import base64
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the autonomous mixer
from AIAudioAgent import AutonomousMixer

app = FastAPI(title="Soundboy AI", version="1.0.0")

# Allow Next.js frontend (localhost:3010 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3010",
        "http://10.0.0.1:3010",
        "https://frontend-three-mu-13.vercel.app",
        "https://soundboy-ai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp storage for mix data (keyed by filename)
mix_data_store: dict = {}

# Source audio store: preserves input audio per filename for chat-mix re-processing
last_processed_input: dict = {}

# Output directory
OUTPUT_DIR = Path(tempfile.gettempdir()) / "soundboy_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Pydantic models ---

class ChatMixRequest(BaseModel):
    message: str
    current_mix: Any
    filename: str


# --- Routes ---

@app.get("/health")
def health():
    return {"ok": True, "status": "live"}


@app.post("/api/mix")
async def mix_audio(file: UploadFile = File(...)):
    """
    Accepts a .wav/.mp3/.flac file, runs the full AI mixing pipeline,
    returns JSON with audio_base64 + mixing_decisions.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    SUPPORTED = (".wav", ".mp3", ".flac")
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in SUPPORTED):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, and .flac files are supported")

    input_path = OUTPUT_DIR / f"input_{file.filename}"
    source_path = OUTPUT_DIR / f"source_{file.filename}"   # permanent copy for chat-mix
    output_path = OUTPUT_DIR / f"mixed_{file.filename}"
    decisions_path = OUTPUT_DIR / f"decisions_{file.filename}.json"

    try:
        # Write uploaded file
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        # Save a permanent copy for chat-mix re-processing
        shutil.copy2(str(input_path), str(source_path))
        last_processed_input[file.filename] = str(source_path)

        # Run the autonomous mixer
        mixer = AutonomousMixer(api_key=api_key)

        import soundfile as sf
        audio_data, sr = mixer._generate_spectrogram(str(input_path))
        mix_instructions = mixer._consult_ai_agent()

        # Save mix decisions
        with open(decisions_path, "w") as f:
            json.dump(mix_instructions, f, indent=2)
        mix_data_store[file.filename] = mix_instructions

        # Apply DSP
        mixer._apply_dsp(audio_data, sr, mix_instructions, str(output_path))

        if os.path.exists("temp_spectrogram.png"):
            os.remove("temp_spectrogram.png")

        # Return JSON with base64 audio + decisions
        with open(str(output_path), "rb") as af:
            audio_b64 = base64.b64encode(af.read()).decode("utf-8")

        return JSONResponse(content={
            "audio_base64": audio_b64,
            "mixing_decisions": mix_instructions,
        })

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up only the temp input (keep source_path for chat-mix)
        if input_path.exists():
            input_path.unlink()


@app.get("/api/mix-data/{filename}")
def get_mix_data(filename: str):
    """Returns the AI mixing decisions for the last processed file."""
    if filename in mix_data_store:
        return JSONResponse(content=mix_data_store[filename])

    decisions_path = OUTPUT_DIR / f"decisions_{filename}.json"
    if decisions_path.exists():
        with open(decisions_path) as f:
            return JSONResponse(content=json.load(f))

    raise HTTPException(status_code=404, detail="No mix data found for this file")


@app.post("/api/chat-mix")
async def chat_mix(req: ChatMixRequest):
    """
    Takes a user message + current mix JSON + filename.
    Calls GPT-5.4 to get an updated mix, re-applies DSP to the source audio,
    returns reply + updated_mix + audio_base64.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    # Locate source audio
    source_path_str = last_processed_input.get(req.filename)
    if not source_path_str:
        # Try disk fallback
        candidate = OUTPUT_DIR / f"source_{req.filename}"
        if candidate.exists():
            source_path_str = str(candidate)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Source audio for '{req.filename}' not found. Please re-upload the file."
            )

    client = openai.OpenAI(api_key=api_key)

    system_prompt = (
        "You are a professional audio engineer AI. The user wants to adjust their mix. "
        "Return JSON only with keys: reply (string), updated_mix (object with same schema as current_mix). "
        "Do not include markdown fences or any text outside the JSON object."
    )
    user_message = f"Current mix: {json.dumps(req.current_mix)}\n\nUser request: {req.message}"

    try:
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT call failed: {e}")

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        gpt_result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"GPT returned invalid JSON: {e}\nRaw: {raw[:500]}")

    reply = gpt_result.get("reply", "Mix updated.")
    updated_mix = gpt_result.get("updated_mix", req.current_mix)

    # Re-apply DSP with updated_mix
    output_path = OUTPUT_DIR / f"chatmix_{req.filename}"
    try:
        mixer = AutonomousMixer(api_key=api_key)
        audio_data, sr = mixer._generate_spectrogram(source_path_str)
        mixer._apply_dsp(audio_data, sr, updated_mix, str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DSP re-apply failed: {e}")

    with open(str(output_path), "rb") as af:
        audio_b64 = base64.b64encode(af.read()).decode("utf-8")

    return JSONResponse(content={
        "reply": reply,
        "updated_mix": updated_mix,
        "audio_base64": audio_b64,
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Soundboy AI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
