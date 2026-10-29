# "Hey Quant" Custom KWS Training Pipeline

Trains a DS-CNN-Small keyword spotter for the phrase "hey astra", then
quantizes it to INT8 TFLite for ESP32-S3 deployment.

## 0. Install deps

```bash
pip install tensorflow numpy librosa soundfile scipy tqdm --break-system-packages
# Optional, for synthetic positive generation:
pip install piper-tts --break-system-packages
```

## 1. Collect raw audio

Put raw (any format/sample-rate) recordings into:

```
raw/
  positive/       # you and others saying "hey astra" — aim for 50-150 clips
  hard_negative/  # phonetically close phrases, deliberately said
  negative/       # unpack Google Speech Commands here (other single words)
  background/     # unpack ESC-50 / MUSAN noise clips here
```

Google Speech Commands: https://www.tensorflow.org/datasets/catalog/speech_commands
ESC-50: https://github.com/karoldvl/ESC-50

Record positives with your phone in several rooms/distances. Vary pace and
loudness. 2-3 seconds per clip, phrase somewhere in the middle.

## 2. (Optional) Generate synthetic positives with Piper TTS

```bash
python generate_synthetic_positives.py --text "hey astra" --n 2000 --out raw/positive_synth
```

This needs Piper voice models downloaded once (script prints the command).

## 3. Standardize + augment + split

```bash
python prepare_dataset.py --raw_dir raw --out_dir dataset
```

Produces `dataset/train`, `dataset/val`, `dataset/test`, each with
`positive/`, `negative/`, `background/` subfolders of 1-second 16kHz mono
WAVs, augmented with noise mixing, time-shift, and pitch jitter.

## 4. Train

```bash
python train.py --data_dir dataset --epochs 40 --out_dir model_out
```

Saves `model_out/kws_model` (SavedModel) and `model_out/best.weights.h5`.

## 5. Quantize to INT8 TFLite

```bash
python quantize.py --saved_model model_out/kws_model --data_dir dataset --out model_out/hey_astra_int8.tflite
```

## 6. Evaluate

```bash
python evaluate.py --tflite model_out/hey_astra_int8.tflite --data_dir dataset/test
```

Reports: accuracy, false-reject rate, false-accept rate, estimated
false-activations/hour, and per-inference latency (host CPU, as a rough
proxy — re-measure on the actual ESP32-S3 before trusting these numbers).

## Notes / things you must verify yourself

- All thresholds (SNR ranges, augmentation counts, debounce frame count)
  are reasonable starting points, NOT tuned values. Tune them against
  your own validation set.
- The <256KB RAM / <10% CPU targets are only meaningful once measured on
  the actual ESP32-S3 — this pipeline gets you a trained + quantized
  model; the on-device benchmarking is a separate, later step.
