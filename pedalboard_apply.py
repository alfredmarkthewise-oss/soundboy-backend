#!/usr/bin/env python3
"""
Pedalboard Audio Effects Processor
Applies mixing decisions from AI analysis to audio files using pedalboard.
"""

import json
import sys
import os
import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,
    Compressor,
    Reverb,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    Gain
)

try:
    from pedalboard import Delay as PedalboardDelay
except ImportError:
    PedalboardDelay = None


def parse_ratio(value) -> float:
    """Parse compressor ratio from various formats.

    Handles:
    - float: 4.0 -> 4.0
    - int: 4 -> 4.0
    - string: "4:1" -> 4.0
    - string: "4" -> 4.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle "4:1" format
        if ":" in value:
            try:
                return float(value.split(":")[0])
            except (ValueError, IndexError):
                return 4.0
        # Handle plain string number
        try:
            return float(value)
        except ValueError:
            return 4.0
    return 4.0  # default


def normalize_eq_entry(eq_entry: dict) -> dict:
    """Normalize EQ entry to internal format.

    Handles both schemas:
    - New: target_frequency_hz, q_factor, gain_db
    - Old: frequency, q, gain_db, band
    """
    normalized = {}

    # Frequency: target_frequency_hz (new) or frequency (old)
    normalized["frequency"] = eq_entry.get("target_frequency_hz", eq_entry.get("frequency", 1000))

    # Q factor: q_factor (new) or q (old)
    normalized["q"] = eq_entry.get("q_factor", eq_entry.get("q", 0.707))

    # Gain: gain_db (both)
    normalized["gain_db"] = eq_entry.get("gain_db", 0)

    # Band type: default to peak
    normalized["band"] = eq_entry.get("band", "peak")

    return normalized


def normalize_schema(decisions: dict) -> dict:
    """Normalize old/new mixing schemas into a unified internal format.

    Handles both schema versions:
    - v1 (legacy): eq, compression, spatial
    - v2 (canonical): subtractive_eq, dynamics.compressor, spatial_effects.delay
    """
    normalized = {
        "eq": [],
        "compression": {},
        "spatial": {},
        "analysis": decisions.get("analysis", {})
    }

    # EQ: old "eq" or new "subtractive_eq"
    raw_eq = decisions.get("eq", decisions.get("subtractive_eq", []))
    if isinstance(raw_eq, list):
        normalized["eq"] = [normalize_eq_entry(eq) for eq in raw_eq]

    # Compression: old "compression" or new "dynamics.compressor"
    raw_compression = {}
    if "compression" in decisions and isinstance(decisions["compression"], dict):
        raw_compression = decisions["compression"]
    else:
        dynamics = decisions.get("dynamics", {})
        if isinstance(dynamics, dict):
            raw_compression = dynamics.get("compressor", {}) or {}

    # Normalize compression settings with ratio parsing
    if raw_compression:
        normalized["compression"] = {
            "threshold_db": raw_compression.get("threshold_db", -20),
            "ratio": parse_ratio(raw_compression.get("ratio", 4.0)),
            "attack_ms": raw_compression.get("attack_ms", 10),
            "release_ms": raw_compression.get("release_ms", 100),
            "makeup_gain_db": raw_compression.get("makeup_gain_db", 0)
        }

    # Spatial: old "spatial" or new "spatial_effects"
    spatial = {}
    if isinstance(decisions.get("spatial"), dict):
        spatial.update(decisions["spatial"])

    spatial_effects = decisions.get("spatial_effects", {})
    if isinstance(spatial_effects, dict):
        reverb = spatial_effects.get("reverb", {})
        if isinstance(reverb, dict):
            if "room_size" in reverb and "reverb_room_size" not in spatial:
                spatial["reverb_room_size"] = reverb["room_size"]
            if "wet" in reverb and "reverb_wet" not in spatial:
                spatial["reverb_wet"] = reverb["wet"]
            if "wet_level" in reverb and "reverb_wet" not in spatial:
                spatial["reverb_wet"] = reverb["wet_level"]
            if "stereo_width" in reverb and "stereo_width" not in spatial:
                spatial["stereo_width"] = reverb["stereo_width"]

        delay = spatial_effects.get("delay", {})
        if isinstance(delay, dict):
            if "time_ms" in delay:
                spatial["delay_ms"] = delay["time_ms"]
            if "mix_percentage" in delay:
                spatial["delay_mix_percentage"] = delay["mix_percentage"]
            if "feedback" in delay:
                spatial["delay_feedback"] = delay["feedback"]

    normalized["spatial"] = spatial
    return normalized


def load_mixing_decisions(json_path: str = "mixing_decisions.json") -> dict:
    """Load mixing decisions from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_eq_chain(eq_settings: list, sample_rate: float) -> list:
    """Create EQ effects chain from mixing decisions."""
    effects = []

    for eq in eq_settings:
        band_type = eq.get("band", "peak")
        freq = eq.get("frequency", 1000)
        gain = eq.get("gain_db", 0)
        q = eq.get("q", 0.707)

        if band_type == "low_shelf":
            effects.append(LowShelfFilter(
                cutoff_frequency_hz=freq,
                gain_db=gain,
                q=q
            ))
        elif band_type == "high_shelf":
            effects.append(HighShelfFilter(
                cutoff_frequency_hz=freq,
                gain_db=gain,
                q=q
            ))
        else:  # peak filter
            effects.append(PeakFilter(
                cutoff_frequency_hz=freq,
                gain_db=gain,
                q=q
            ))

    return effects


def create_compressor(comp_settings: dict) -> Compressor:
    """Create compressor from mixing decisions."""
    return Compressor(
        threshold_db=comp_settings.get("threshold_db", -20),
        ratio=comp_settings.get("ratio", 4.0),
        attack_ms=comp_settings.get("attack_ms", 10),
        release_ms=comp_settings.get("release_ms", 100)
    )


def create_reverb(spatial_settings: dict) -> Reverb:
    """Create reverb from spatial settings."""
    return Reverb(
        room_size=spatial_settings.get("reverb_room_size", 0.3),
        wet_level=spatial_settings.get("reverb_wet", 0.15),
        dry_level=1.0 - spatial_settings.get("reverb_wet", 0.15),
        width=min(1.0, spatial_settings.get("stereo_width", 1.0))
    )


def apply_delay_effect(audio: np.ndarray, sample_rate: float, time_ms: float, mix_percentage: float) -> np.ndarray:
    """Apply delay effect using pedalboard Delay when available, else numpy fallback."""
    if time_ms <= 0 or mix_percentage <= 0:
        return audio

    mix = max(0.0, min(100.0, mix_percentage)) / 100.0
    if mix <= 0:
        return audio

    if PedalboardDelay is not None:
        try:
            delay_board = Pedalboard([
                PedalboardDelay(delay_seconds=float(time_ms) / 1000.0, feedback=0.0, mix=mix)
            ])
            return delay_board(audio, sample_rate)
        except Exception:
            pass

    delay_samples = int(sample_rate * (float(time_ms) / 1000.0))
    if delay_samples <= 0 or delay_samples >= len(audio):
        return audio

    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples]
    return (audio * (1.0 - mix)) + (delayed * mix)


def apply_effects(
    input_path: str,
    output_path: str,
    decisions: dict
) -> dict:
    """Apply all effects from mixing decisions to audio file."""
    audio, sample_rate = sf.read(input_path)

    # Handle mono files
    if len(audio.shape) == 1:
        audio = audio.reshape(-1, 1)

    effects = []
    applied_summary = {
        "eq_bands": 0,
        "compression": False,
        "reverb": False,
        "delay": False,
        "delay_ms": 0,
        "delay_mix_percentage": 0,
        "makeup_gain": 0
    }

    # Add EQ effects
    if "eq" in decisions:
        eq_effects = create_eq_chain(decisions["eq"], sample_rate)
        effects.extend(eq_effects)
        applied_summary["eq_bands"] = len(eq_effects)

    # Add compressor
    if "compression" in decisions:
        effects.append(create_compressor(decisions["compression"]))
        applied_summary["compression"] = True

        # Add makeup gain if specified
        makeup = decisions["compression"].get("makeup_gain_db", 0)
        if makeup != 0:
            effects.append(Gain(gain_db=makeup))
            applied_summary["makeup_gain"] = makeup

    # Add reverb
    if "spatial" in decisions:
        spatial = decisions["spatial"]
        if spatial.get("reverb_wet", 0) > 0:
            effects.append(create_reverb(spatial))
            applied_summary["reverb"] = True

    # Create and apply pedalboard
    board = Pedalboard(effects)
    processed = board(audio, sample_rate)

    # Add delay (supports old "spatial.delay_ms" and new "spatial_effects.delay.time_ms")
    spatial = decisions.get("spatial", {})
    delay_ms = spatial.get("delay_ms", 0)
    delay_mix = spatial.get("delay_mix_percentage", 25 if delay_ms else 0)
    processed = apply_delay_effect(processed, sample_rate, delay_ms, delay_mix)
    if delay_ms and delay_mix:
        applied_summary["delay"] = True
        applied_summary["delay_ms"] = delay_ms
        applied_summary["delay_mix_percentage"] = delay_mix

    # Save processed audio
    sf.write(output_path, processed, int(sample_rate))

    return applied_summary


def print_summary(summary: dict, decisions: dict) -> None:
    """Print applied effects summary."""
    print("\n" + "=" * 50)
    print("APPLIED EFFECTS SUMMARY")
    print("=" * 50)

    if summary["eq_bands"] > 0:
        print(f"\nEQ ({summary['eq_bands']} bands):")
        for eq in decisions.get("eq", []):
            print(f"  - {eq['band']}: {eq['frequency']}Hz @ {eq['gain_db']:+.1f}dB (Q={eq['q']})")

    if summary["compression"]:
        comp = decisions.get("compression", {})
        print(f"\nCompression:")
        print(f"  - Threshold: {comp.get('threshold_db', 0)}dB")
        print(f"  - Ratio: {comp.get('ratio', 4)}:1")
        print(f"  - Attack: {comp.get('attack_ms', 10)}ms")
        print(f"  - Release: {comp.get('release_ms', 100)}ms")
        if summary["makeup_gain"]:
            print(f"  - Makeup Gain: {summary['makeup_gain']:+.1f}dB")

    if summary["reverb"]:
        spatial = decisions.get("spatial", {})
        print(f"\nReverb:")
        print(f"  - Room Size: {spatial.get('reverb_room_size', 0.3):.1%}")
        print(f"  - Wet Level: {spatial.get('reverb_wet', 0.15):.1%}")
        print(f"  - Stereo Width: {spatial.get('stereo_width', 1.0):.2f}")

    if summary.get("delay"):
        print(f"\nDelay:")
        print(f"  - Time: {summary.get('delay_ms', 0)}ms")
        print(f"  - Mix: {summary.get('delay_mix_percentage', 0)}%")

    if "analysis" in decisions:
        analysis = decisions["analysis"]
        print(f"\nAnalysis:")
        print(f"  - Source Type: {analysis.get('source_type', 'unknown')}")
        print(f"  - Key Issues: {', '.join(analysis.get('key_issues', []))}")
        print(f"  - Confidence: {analysis.get('confidence', 0):.0%}")

    print("\n" + "=" * 50)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.wav"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "processed_output.wav"
    decisions_path = sys.argv[3] if len(sys.argv) > 3 else "mixing_decisions.json"

    if not os.path.exists(input_path):
        print(f"Error: Input audio not found at {input_path}")
        sys.exit(1)

    if not os.path.exists(decisions_path):
        print(f"Error: Mixing decisions not found at {decisions_path}")
        print("Run audio_engineer_agent.py first to generate mixing decisions.")
        sys.exit(1)

    print(f"Loading audio: {input_path}")
    print(f"Loading decisions: {decisions_path}")

    decisions = normalize_schema(load_mixing_decisions(decisions_path))

    print("Applying effects...")
    summary = apply_effects(input_path, output_path, decisions)

    print(f"\nProcessed audio saved to: {output_path}")
    print_summary(summary, decisions)


if __name__ == "__main__":
    main()
