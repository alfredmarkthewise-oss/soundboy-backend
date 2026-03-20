"""
Soundboy AI — FastAPI Backend Server
Serves /api/mix (POST) and /api/mix-data (GET) for the Next.js frontend.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import the autonomous mixer
from AIAudioAgent import AutonomousMixer

app = FastAPI(title="Soundboy AI", version="1.0.0")

# Allow Next.js frontend (localhost:3010 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your Vercel URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp storage for mix data (keyed by filename)
mix_data_store: dict = {}

# Output directory
OUTPUT_DIR = Path(tempfile.gettempdir()) / "soundboy_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"ok": True, "status": "live"}


@app.post("/api/mix")
async def mix_audio(file: UploadFile = File(...)):
    """
    Accepts a .wav file, runs the full AI mixing pipeline,
    returns the processed .wav file as a download.
    """
    # Validate API key — check multiple possible secret names
    api_key = (
        os.environ.get("OPENAI_API_KEY") or
        os.environ.get("Open_AI_Key") or
        os.environ.get("OPEN_AI_KEY") or
        os.environ.get("openai_api_key")
    )
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    SUPPORTED = (".wav", ".mp3", ".flac")
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in SUPPORTED):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, and .flac files are supported")

    # Save upload to temp
    # Always save input with original extension, output as wav
    safe_name = file.filename.replace(" ", "_")
    input_path = OUTPUT_DIR / f"input_{safe_name}"
    output_path = OUTPUT_DIR / f"mixed_{safe_name}.wav"
    decisions_path = OUTPUT_DIR / f"decisions_{safe_name}.json"

    try:
        # Write uploaded file
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run the autonomous mixer
        mixer = AutonomousMixer(api_key=api_key)

        # Patch to also save decisions
        import soundfile as sf
        audio_data, sr = mixer._generate_spectrogram(str(input_path))
        mix_instructions = mixer._consult_ai_agent()

        # Save mix decisions for the frontend
        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump(mix_instructions, f, indent=2, ensure_ascii=False)

        # Store in memory for GET endpoint
        mix_data_store[file.filename] = mix_instructions

        # Apply DSP
        mixer._apply_dsp(audio_data, sr, mix_instructions, str(output_path))

        # Clean up mixer's temp spectrogram
        if os.path.exists("temp_spectrogram.png"):
            os.remove("temp_spectrogram.png")

        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=f"AI_Mixed_{file.filename}",
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file
        if input_path.exists():
            input_path.unlink()


@app.get("/api/mix-data/{filename}")
def get_mix_data(filename: str):
    """Returns the AI mixing decisions for the last processed file."""
    if filename in mix_data_store:
        return JSONResponse(content=mix_data_store[filename])

    # Try from disk
    decisions_path = OUTPUT_DIR / f"decisions_{filename}.json"
    if decisions_path.exists():
        with open(decisions_path, encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))

    raise HTTPException(status_code=404, detail="No mix data found for this file")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Soundboy AI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
