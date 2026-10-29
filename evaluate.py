"""
Evaluates the quantized TFLite model on the held-out test set.
"""
import argparse
import glob
import os
import time

import numpy as np
import soundfile as sf
import tensorflow as tf

from features import log_mel

CLASSES = ["background", "negative", "positive"]
POSITIVE_IDX = CLASSES.index("positive")


def load_test_set(data_dir):
    items = []
    for idx, cls in enumerate(CLASSES):
        for f in glob.glob(os.path.join(data_dir, cls, "*.wav")):
            items.append((f, idx))
    return items


def run_inference(interpreter, feat, in_scale, in_zero):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    feat_q = (feat / in_scale + in_zero).astype(np.int8)
    feat_q = feat_q[np.newaxis, ..., np.newaxis]

    interpreter.set_tensor(input_details["index"], feat_q)
    t0 = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    out = interpreter.get_tensor(output_details["index"])
    out_scale, out_zero = output_details["quantization"]
    probs = (out.astype(np.float32) - out_zero) * out_scale
    return probs[0], latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    in_scale, in_zero = input_details["quantization"]

    items = load_test_set(args.data_dir)
    print(f"Evaluating on {len(items)} test clips")

    y_true, y_pred, latencies = [], [], []
    for path, label in items:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        feat = log_mel(y.astype(np.float32))
        probs, latency_ms = run_inference(interpreter, feat, in_scale, in_zero)
        y_true.append(label)
        y_pred.append(int(np.argmax(probs)))
        latencies.append(latency_ms)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    pos_mask = y_true == POSITIVE_IDX
    tp = np.sum((y_true == POSITIVE_IDX) & (y_pred == POSITIVE_IDX))
    fn = np.sum((y_true == POSITIVE_IDX) & (y_pred != POSITIVE_IDX))
    frr = fn / max(pos_mask.sum(), 1)
    neg_mask = y_true != POSITIVE_IDX
    fp = np.sum((y_true != POSITIVE_IDX) & (y_pred == POSITIVE_IDX))
    far = fp / max(neg_mask.sum(), 1)

    inferences_per_hour = 3600 / 0.01
    est_far_per_hour = far * inferences_per_hour

    print(f"Overall accuracy:        {acc*100:.2f}%")
    print(f"True positive rate:      {(tp/max(pos_mask.sum(),1))*100:.2f}%")
    print(f"False reject rate:       {frr*100:.2f}%")
    print(f"False accept rate:       {far*100:.4f}%  (per single inference)")
    print(f"Est. false activations/hour (RAW, before debounce logic): {est_far_per_hour:.1f}")
    print(f"Mean host-CPU latency:   {np.mean(latencies):.3f} ms")


if __name__ == "__main__":
    main()
