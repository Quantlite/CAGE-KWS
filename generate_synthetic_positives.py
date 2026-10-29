"""
Generate synthetic "hey astra" positive clips using Piper TTS (open-source,
offline, Apache-2.0). This mirrors the technique openWakeWord uses to
bootstrap positive data cheaply when real recordings are scarce.

Piper ships many voice models (different speakers/accents). Using several
voices + speed/pitch jitter gives you speaker diversity without recording
hundreds of humans.

Usage:
    python generate_synthetic_positives.py --text "hey astra" --n 2000 --out raw/positive_synth
"""
import argparse
import os
import random
import subprocess
import sys
import urllib.request

# Voice name -> Hugging Face path fragment (rhasspy/piper-voices repo layout)
VOICE_PATHS = {
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-amy-medium": "en/en_US/amy/medium",
    "en_US-ryan-medium": "en/en_US/ryan/medium",
    "en_GB-alan-medium": "en/en_GB/alan/medium",
    "en_GB-jenny_dioco-medium": "en/en_GB/jenny_dioco/medium",
}
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def ensure_piper_installed():
    try:
        result = subprocess.run(["piper", "--version"], capture_output=True, check=False)
        if result.returncode != 0:
            print("piper CLI found but returned an error on --version:", file=sys.stderr)
            print(result.stderr.decode(errors="ignore"), file=sys.stderr)
    except FileNotFoundError:
        print("Piper not found on PATH. Install with: pip install piper-tts", file=sys.stderr)
        sys.exit(1)


def download_voice(voice: str, models_dir: str) -> str:
    """Downloads the .onnx + .onnx.json for a voice if not already present.
    Returns the local path to the .onnx file (what --model expects)."""
    os.makedirs(models_dir, exist_ok=True)
    onnx_path = os.path.join(models_dir, f"{voice}.onnx")
    json_path = os.path.join(models_dir, f"{voice}.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(json_path):
        return onnx_path

    frag = VOICE_PATHS[voice]
    for fname, local_path in ((f"{voice}.onnx", onnx_path), (f"{voice}.onnx.json", json_path)):
        url = f"{HF_BASE}/{frag}/{fname}"
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}", file=sys.stderr)
            raise
    return onnx_path


def synthesize(text: str, model_path: str, out_path: str, length_scale: float):
    """length_scale >1 = slower speech, <1 = faster. Adds pace diversity."""
    cmd = [
        "piper",
        "--model", model_path,
        "--output_file", out_path,
        "--length_scale", str(length_scale),
    ]
    result = subprocess.run(cmd, input=text.encode(), capture_output=True)
    if result.returncode != 0:
        # Surface the REAL error instead of swallowing it.
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout, stderr=result.stderr,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="hey astra")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="raw/positive_synth")
    ap.add_argument("--models_dir", default="piper_voices")
    args = ap.parse_args()

    ensure_piper_installed()
    os.makedirs(args.out, exist_ok=True)

    # Download every voice ONCE up front, so failures are visible immediately
    # instead of retried (and failing) on every single one of --n clips.
    voice_names = list(VOICE_PATHS.keys())
    model_paths = {}
    for v in voice_names:
        try:
            model_paths[v] = download_voice(v, args.models_dir)
        except Exception:
            print(f"Skipping voice {v} entirely (download failed)", file=sys.stderr)
    if not model_paths:
        print("No voices downloaded successfully. Cannot continue.", file=sys.stderr)
        sys.exit(1)

    available_voices = list(model_paths.keys())
    fail_count = 0
    for i in range(args.n):
        voice = random.choice(available_voices)
        length_scale = random.uniform(0.85, 1.2)  # pace jitter
        out_path = os.path.join(args.out, f"synth_{i:05d}_{voice}.wav")
        try:
            synthesize(args.text, model_paths[voice], out_path, length_scale)
        except subprocess.CalledProcessError as e:
            fail_count += 1
            if fail_count <= 3:  # only print the first few in full, to avoid spam
                print(f"Failed on {voice}:", file=sys.stderr)
                print(e.stderr.decode(errors="ignore"), file=sys.stderr)
            continue
        if i % 100 == 0:
            print(f"[{i}/{args.n}] generated")

    if fail_count:
        print(f"\n{fail_count}/{args.n} clips failed. See errors above.", file=sys.stderr)

    print(f"Done. {args.n} synthetic clips in {args.out}")
    print("NOTE: synthetic clips alone will NOT generalize to your real mic/room.")
    print("They supplement, not replace, real recordings.")


if __name__ == "__main__":
    main()
