"""
Live microphone test for the "hey astra" KWS model.

Continuously listens through your laptop mic, runs the same feature
extraction + quantized model used in training, and prints a live
probability readout plus a DETECTED banner when the keyword is heard.

This is the first real-world sanity check with your actual voice, room,
and mic -- everything before this only tested against pre-recorded
dataset clips.

Install (one-time):
    pip install --user sounddevice
    sudo apt install -y libportaudio2   # system audio library sounddevice needs

Usage:
    python3 live_test.py --tflite model_out/hey_astra_int8.tflite
"""
import argparse
import collections
import sys
import time

import numpy as np
import sounddevice as sd
import tensorflow as tf

from features import log_mel, feature_shape

SR = 16000
CLIP_LEN = SR  # 1 second sliding window
STRIDE_SAMPLES = int(SR * 0.1)  # run inference every 100ms (10 inferences/sec)
CLASSES = ["background", "negative", "positive"]
POSITIVE_IDX = CLASSES.index("positive")


def load_interpreter(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, feat):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    feat_q = (feat / in_scale + in_zero).astype(np.int8)
    feat_q = feat_q[np.newaxis, ..., np.newaxis]

    interpreter.set_tensor(input_details["index"], feat_q)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])
    probs = (out.astype(np.float32) - out_zero) * out_scale
    return probs[0]


def print_bar(probs, detected):
    labels = ["bg", "neg", "POS"]
    bars = []
    for label, p in zip(labels, probs):
        n_blocks = int(p * 20)
        bars.append(f"{label}:{'#'*n_blocks}{'.'*(20-n_blocks)} {p:.2f}")
    line = "  ".join(bars)
    marker = "  <<< DETECTED! >>>" if detected else ""
    sys.stdout.write(f"\r{line}{marker}    ")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="model_out/hey_astra_int8.tflite")
    ap.add_argument("--threshold", type=float, default=0.85,
                     help="Probability threshold for the positive class")
    ap.add_argument("--consecutive", type=int, default=3,
                     help="Number of consecutive windows above threshold "
                          "required to count as a real detection (debounce)")
    ap.add_argument("--cooldown_s", type=float, default=1.5,
                     help="Seconds to wait after a detection before allowing another")
    args = ap.parse_args()

    print(f"Loading model: {args.tflite}")
    interpreter = load_interpreter(args.tflite)

    audio_buffer = collections.deque(maxlen=CLIP_LEN)
    audio_buffer.extend(np.zeros(CLIP_LEN, dtype=np.float32))

    consecutive_hits = 0
    last_detection_time = 0
    new_samples = collections.deque()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio status] {status}", file=sys.stderr)
        new_samples.extend(indata[:, 0].copy())

    print(f"Listening... say 'hey astra' near your mic. Ctrl+C to stop.")
    print(f"Threshold={args.threshold}, consecutive={args.consecutive}, "
          f"cooldown={args.cooldown_s}s\n")

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                             blocksize=STRIDE_SAMPLES, callback=callback):
            while True:
                # Wait until we have a new stride's worth of audio
                while len(new_samples) < STRIDE_SAMPLES:
                    time.sleep(0.01)

                chunk = [new_samples.popleft() for _ in range(STRIDE_SAMPLES)]
                audio_buffer.extend(chunk)

                y = np.array(audio_buffer, dtype=np.float32)
                feat = log_mel(y)
                probs = run_inference(interpreter, feat)

                now = time.time()
                is_hit = probs[POSITIVE_IDX] > args.threshold
                if is_hit:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                detected = False
                if (consecutive_hits >= args.consecutive
                        and now - last_detection_time > args.cooldown_s):
                    detected = True
                    last_detection_time = now
                    consecutive_hits = 0

                print_bar(probs, detected)
                if detected:
                    print()  # newline so the DETECTED banner isn't overwritten

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
