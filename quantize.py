"""
Converts the trained SavedModel to a fully INT8-quantized TFLite model
suitable for TensorFlow Lite for Microcontrollers on ESP32-S3.

Full-integer quantization (weights AND activations INT8) is required --
not just weight quantization -- because TFLM's reference/ESP-NN kernels
expect INT8 activations to get the CPU/latency benefit. This is also why
a representative dataset is mandatory: it's used to calibrate activation
ranges per-layer.

Usage:
    python quantize.py --saved_model model_out/kws_model --data_dir dataset \
        --out model_out/hey_astra_int8.tflite
"""
import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf
import tensorflow as tf

from features import log_mel, feature_shape


def representative_dataset_gen(data_dir, n_samples=300):
    files = glob.glob(os.path.join(data_dir, "train", "*", "*.wav"))
    random.shuffle(files)
    files = files[:n_samples]
    for f in files:
        y, sr = sf.read(f)
        if y.ndim > 1:
            y = y.mean(axis=1)
        feat = log_mel(y.astype(np.float32))
        feat = np.expand_dims(feat, (0, -1)).astype(np.float32)  # (1, frames, mels, 1)
        yield [feat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model", default="model_out/kws_model")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out", default="model_out/hey_astra_int8.tflite")
    args = ap.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(args.data_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Wrote {args.out}  ({size_kb:.1f} KB)")
    print("Next: run evaluate.py on the held-out test set to check accuracy "
          "did not meaningfully drop after quantization (compare against the "
          "float model before trusting this).")


if __name__ == "__main__":
    main()
