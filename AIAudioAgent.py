import os
import json
import base64
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, no font cache issues
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # ASCII-safe font
import numpy as np
import soundfile as sf
import requests
from pedalboard import Pedalboard, Compressor, PeakFilter, Delay

class AutonomousMixer:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it via environment variable or pass directly.")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.spectrogram_path = "temp_spectrogram.png"

    def _generate_spectrogram(self, audio_path):
        print("1. Extracting acoustic data and generating Mel-spectrogram...")
        y, sr = librosa.load(audio_path, sr=48000)

        # Calculate the math: STFT to Mel Scale
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Save the visual matrix
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512)
        plt.tight_layout(pad=0)
        plt.savefig(self.spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return y, sr

    def _consult_ai_agent(self):
        print("2. Sending matrix to AI for DSP analysis...")
        with open(self.spectrogram_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        system_prompt = """You are a world-class DSP audio engineer and mixing specialist.
You will receive a Mel-spectrogram image of an audio recording.
Your ONLY job is to return a JSON object with specific mixing instructions.
You MUST return ONLY raw JSON — no markdown, no code blocks, no explanation, no text before or after.
Start your response with { and end with }."""

        user_prompt = """Analyze this Mel-spectrogram and return mixing decisions.

You MUST return ONLY this JSON structure (no markdown, no backticks, raw JSON only):
{
    "subtractive_eq": [
        {"target_frequency_hz": 250, "q_factor": 1.5, "gain_db": -3.5, "reason": "low-mid buildup"},
        {"target_frequency_hz": 4000, "q_factor": 3.0, "gain_db": -2.0, "reason": "harsh resonance"}
    ],
    "dynamics": {
        "compressor": {
            "threshold_db": -18,
            "ratio": 4.0,
            "attack_ms": 15,
            "release_ms": 80,
            "makeup_gain_db": 2.5,
            "intent": "control dynamics"
        }
    },
    "spatial_effects": {
        "delay": {
            "time_ms": 20,
            "feedback": 0,
            "mix_percentage": 15,
            "intent": "add presence"
        }
    }
}

Adjust the values based on what you see in the spectrogram. Return ONLY the JSON object."""

        payload = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        resp_data = response.json()

        if "error" in resp_data:
            raise RuntimeError(f"OpenAI API error: {resp_data['error']}")

        raw_output = resp_data['choices'][0]['message']['content']

        # Belt-and-suspenders: strip any accidental markdown
        clean_json = raw_output.strip()
        if clean_json.startswith("```"):
            lines = clean_json.split("\n")
            clean_json = "\n".join(lines[1:-1]).strip()

        return json.loads(clean_json)

    def _parse_ratio(self, value) -> float:
        """Parse compressor ratio from various formats (e.g., 4.0 or '4:1')."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if ":" in value:
                try:
                    return float(value.split(":")[0])
                except (ValueError, IndexError):
                    return 4.0
            try:
                return float(value)
            except ValueError:
                return 4.0
        return 4.0

    def _apply_dsp(self, audio_data, sample_rate, mix_instructions, output_path):
        print("3. Executing mathematical DSP instructions...")
        plugins = []

        # A. Apply Subtractive EQ
        # NOTE: Pedalboard uses PeakFilter with cutoff_frequency_hz (not PeakingFilter)
        for eq in mix_instructions.get("subtractive_eq", []):
            plugins.append(PeakFilter(
                cutoff_frequency_hz=eq["target_frequency_hz"],
                q=eq["q_factor"],
                gain_db=eq["gain_db"]
            ))

        # B. Apply Compressor
        if "dynamics" in mix_instructions and "compressor" in mix_instructions["dynamics"]:
            comp = mix_instructions["dynamics"]["compressor"]
            ratio = self._parse_ratio(comp["ratio"])
            plugins.append(Compressor(
                threshold_db=comp["threshold_db"],
                ratio=ratio,
                attack_ms=comp["attack_ms"],
                release_ms=comp["release_ms"]
            ))

        # C. Apply Delay (Space)
        if "spatial_effects" in mix_instructions and "delay" in mix_instructions["spatial_effects"]:
            delay = mix_instructions["spatial_effects"]["delay"]
            # Normalize mix_percentage: API may send 0-100 or 0-1
            mix_pct = float(delay.get("mix_percentage", 15))
            mix_val = mix_pct / 100.0 if mix_pct > 1.0 else mix_pct
            # Normalize feedback: clamp to 0.0-1.0, handle percentages
            fb = float(delay.get("feedback", 0.0))
            fb = fb / 100.0 if fb > 1.0 else fb
            fb = max(0.0, min(1.0, fb))
            plugins.append(Delay(
                delay_seconds=max(0.001, float(delay.get("time_ms", 20)) / 1000.0),
                feedback=fb,
                mix=max(0.0, min(1.0, mix_val))
            ))

        # Build and run the virtual console
        board = Pedalboard(plugins)
        processed_audio = board(audio_data, sample_rate)

        # Apply makeup gain mathematically
        if "dynamics" in mix_instructions and "compressor" in mix_instructions["dynamics"]:
            makeup_db = mix_instructions["dynamics"]["compressor"].get("makeup_gain_db", 0)
            processed_audio *= (10 ** (makeup_db / 20))

        sf.write(output_path, processed_audio, sample_rate)
        print(f"Done! Mixed file saved as: {output_path}")

    def process_file(self, input_wav, output_wav):
        """The main execution loop"""
        audio_data, sr = self._generate_spectrogram(input_wav)
        mix_instructions = self._consult_ai_agent()
        print(f"AI Decisions: {json.dumps(mix_instructions, indent=2)}")
        self._apply_dsp(audio_data, sr, mix_instructions, output_wav)

        # Clean up temporary visual matrix
        if os.path.exists(self.spectrogram_path):
            os.remove(self.spectrogram_path)

# ==========================================
# RUNNING THE AGENT
# ==========================================
if __name__ == "__main__":
    # Ensure your API key is set in your environment variables
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    agent = AutonomousMixer(api_key=api_key)

    # Drop your raw file in the same folder and run!
    agent.process_file("raw_vocal.wav", "agent_mixed_vocal.wav")
