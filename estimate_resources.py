"""
Estimates RAM footprint and CPU duty cycle for the quantized TFLite model.

IMPORTANT — read this before trusting any number this prints:

These are ESTIMATES from running on your laptop CPU, not measurements from
the actual ESP32-S3. They tell you roughly whether you're in the right
ballpark, not whether you satisfy the PS's <256KB RAM / <10% CPU
requirements. The only way to get real, submittable numbers is flashing
this model to actual ESP32-S3 hardware and measuring there -- host-CPU
inference speed is not comparable to a microcontroller's (different
architecture, no SIMD acceleration from ESP-NN, different clock speed).

What this script actually gives you:
    - Flash footprint: exact (model file size -- this transfers directly to device)
    - RAM (tensor arena) estimate: approximate, from summing declared
      tensor buffer sizes in the TFLite graph. Real TFLM arena usage can
      differ (allocator overhead, buffer reuse patterns) -- treat this as
      a starting estimate to catch obviously-oversized models, not a
      final number.
    - CPU duty cycle estimate: HOST latency divided by assumed on-device
      inference stride. This number is NOT transferable to the ESP32-S3
      at all -- it only tells you the shape of the tradeoff (fewer/lighter
      layers = lower duty cycle), not real percentages.

Usage:
    python3 estimate_resources.py --tflite model_out/hey_astra_int8.tflite
"""
import argparse
import os
import time

import numpy as np
import tensorflow as tf


def estimate_ram(interpreter):
    """Sums declared tensor buffer sizes as a rough RAM proxy.
    Real TFLM arena size depends on the allocator's buffer-reuse strategy
    (intermediate tensors can share memory when their lifetimes don't
    overlap), so this is an UPPER-BOUND-ish estimate, not exact."""
    total_bytes = 0
    tensor_details = interpreter.get_tensor_details()
    for t in tensor_details:
        try:
            arr = interpreter.get_tensor(t["index"])
            total_bytes += arr.nbytes
        except Exception:
            # Some tensors (e.g. unallocated intermediates) may not be
            # readable directly -- estimate from shape/dtype instead.
            shape = t.get("shape", [])
            dtype_size = np.dtype(t["dtype"]).itemsize if "dtype" in t else 1
            n = int(np.prod(shape)) if len(shape) else 0
            total_bytes += n * dtype_size
    return total_bytes


def measure_latency(interpreter, n_runs=200):
    input_details = interpreter.get_input_details()[0]
    shape = input_details["shape"]
    dtype = input_details["dtype"]

    if dtype == np.int8:
        dummy = np.random.randint(-128, 127, size=shape, dtype=np.int8)
    else:
        dummy = np.random.randn(*shape).astype(dtype)

    # Warmup (first call includes allocation overhead, not representative)
    for _ in range(10):
        interpreter.set_tensor(input_details["index"], dummy)
        interpreter.invoke()

    latencies = []
    for _ in range(n_runs):
        interpreter.set_tensor(input_details["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    return np.array(latencies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--stride_ms", type=float, default=10.0,
                     help="Assumed on-device inference stride in ms (how often "
                          "a new inference runs during continuous listening). "
                          "Matches FRAME_STRIDE in features.py by default.")
    args = ap.parse_args()

    size_bytes = os.path.getsize(args.tflite)
    print(f"=== Flash footprint (exact, transfers directly to device) ===")
    print(f"Model file size: {size_bytes/1024:.1f} KB\n")

    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()

    ram_bytes = estimate_ram(interpreter)
    print(f"=== RAM estimate (rough -- see script docstring for caveats) ===")
    print(f"Sum of tensor buffers: {ram_bytes/1024:.1f} KB")
    print(f"NOTE: does not include TFLM interpreter overhead, audio ring buffer,")
    print(f"feature buffer, or network buffers -- see the RAM budget table from")
    print(f"the architecture design (audio buffer ~12.8KB, feature buffer ~4KB,")
    print(f"app/RTOS overhead ~20KB, etc). Add those separately for a full picture.\n")

    latencies = measure_latency(interpreter)
    print(f"=== Host-CPU inference latency (NOT representative of ESP32-S3) ===")
    print(f"Mean:   {latencies.mean():.3f} ms")
    print(f"Median: {np.median(latencies):.3f} ms")
    print(f"P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"Min:    {latencies.min():.3f} ms\n")

    duty_cycle_pct = (latencies.mean() / args.stride_ms) * 100
    print(f"=== Estimated CPU duty cycle (HOST numbers -- illustrative only) ===")
    print(f"Assumed inference stride: {args.stride_ms} ms")
    print(f"Estimated duty cycle: {duty_cycle_pct:.2f}%")
    print(f"(= mean_latency / stride_ms -- i.e. what fraction of each stride")
    print(f"window is spent computing, if this ran continuously)")
    print()
    print("=" * 70)
    print("REMINDER: none of the RAM/CPU numbers above are submittable PS")
    print("evidence. They're sanity checks to catch an obviously oversized")
    print("model before you spend time on hardware bring-up. Real numbers")
    print("must come from flashing this .tflite to the ESP32-S3 and measuring")
    print("with ESP-IDF's heap/CPU profiling (heap_caps_get_free_size(),")
    print("esp_timer_get_time() around the inference call, and idle-task")
    print("CPU% from FreeRTOS runtime stats).")
    print("=" * 70)


if __name__ == "__main__":
    main()
