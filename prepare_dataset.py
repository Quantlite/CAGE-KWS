"""
Standardizes raw audio (any sample rate/format) into 16kHz mono 1-second
WAVs, applies augmentation (noise mixing, time-shift, pitch jitter), and
splits into train/val/test.

Classes:
    positive      -> "hey astra" (real + synthetic)
    negative      -> everything else spoken, incl. hard negatives
    background    -> pure noise, no speech

Usage:
    python prepare_dataset.py --raw_dir raw --out_dir dataset
"""
import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf
import librosa

SR = 16000
CLIP_SECONDS = 1.0
CLIP_LEN = int(SR * CLIP_SECONDS)

# Augmentation ranges — starting points, tune against your validation results.
SNR_DB_RANGE = (0, 20)        # noise-mixing strength for positives/negatives
TIME_SHIFT_MAX = 0.15         # seconds, random shift so keyword isn't always centered
PITCH_STEPS = (-2, 2)         # semitone jitter
N_AUGS_PER_REAL_POSITIVE = 20  # real clips are scarce -> augment heavily
N_AUGS_PER_SYNTH_POSITIVE = 3
N_AUGS_PER_NEGATIVE = 2


def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y


def fit_to_length(y, length=CLIP_LEN):
    """Center-pad or randomly-crop to exactly `length` samples."""
    if len(y) < length:
        pad = length - len(y)
        left = random.randint(0, pad)
        y = np.pad(y, (left, pad - left))
    elif len(y) > length:
        start = random.randint(0, len(y) - length)
        y = y[start:start + length]
    return y


def random_time_shift(y, max_shift=TIME_SHIFT_MAX):
    shift = int(random.uniform(-max_shift, max_shift) * SR)
    return np.roll(y, shift)


def random_pitch_shift(y):
    steps = random.uniform(*PITCH_STEPS)
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)


def mix_noise(y, noise_clips, snr_db_range=SNR_DB_RANGE):
    if not noise_clips:
        return y
    noise = random.choice(noise_clips)
    noise = fit_to_length(noise, len(y))
    sig_power = np.mean(y ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_db = random.uniform(*snr_db_range)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    mixed = y + noise
    peak = np.max(np.abs(mixed)) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def augment_clip(y, noise_clips, do_pitch=True):
    y = random_time_shift(y)
    if do_pitch and random.random() < 0.5:
        y = random_pitch_shift(y)
    y = mix_noise(y, noise_clips)
    return fit_to_length(y)


def collect_wavs(folder):
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
    return files


def split_list(items, ratios=(0.8, 0.1, 0.1)):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def save_clip(y, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, name), y, SR, subtype="PCM_16")


def process_class(files, noise_clips, out_root, class_name, n_augs, is_synth_flags=None):
    """Loads each file, fits/augments, writes multiple augmented copies,
    then splits the resulting AUGMENTED pool into train/val/test.
    (Splitting after augmentation is fine here because augmentation is
    randomized per-copy, not duplicating identical val/test leakage risk
    the way splitting raw files after would avoid — for max rigor, split
    raw files first, then augment each split independently. This script
    does the rigorous version below.)
    """
    raw_train, raw_val, raw_test = split_list(files)
    subsets = {"train": raw_train, "val": raw_val, "test": raw_test}

    for subset_name, subset_files in subsets.items():
        idx = 0
        for f in subset_files:
            try:
                y = load_audio(f)
            except Exception as e:
                print(f"skip {f}: {e}")
                continue
            y0 = fit_to_length(y)
            save_clip(y0, os.path.join(out_root, subset_name, class_name), f"{class_name}_{idx:05d}_orig.wav")
            idx += 1
            n_aug = n_augs if subset_name == "train" else max(1, n_augs // 4)
            for a in range(n_aug):
                ya = augment_clip(y, noise_clips)
                save_clip(ya, os.path.join(out_root, subset_name, class_name), f"{class_name}_{idx:05d}_aug{a}.wav")
                idx += 1
        print(f"{class_name}/{subset_name}: {idx} clips written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="raw")
    ap.add_argument("--out_dir", default="dataset")
    args = ap.parse_args()

    random.seed(42)

    noise_files = collect_wavs(os.path.join(args.raw_dir, "background"))
    noise_clips = [load_audio(f) for f in noise_files]
    print(f"Loaded {len(noise_clips)} background noise clips")

    positive_files = collect_wavs(os.path.join(args.raw_dir, "positive")) + \
        collect_wavs(os.path.join(args.raw_dir, "positive_synth"))
    negative_files = collect_wavs(os.path.join(args.raw_dir, "negative")) + \
        collect_wavs(os.path.join(args.raw_dir, "hard_negative"))
    background_files = noise_files  # background class also trains directly on raw noise

    print(f"positive raw: {len(positive_files)}, negative raw: {len(negative_files)}, "
          f"background raw: {len(background_files)}")

    if len(positive_files) < 20:
        print("WARNING: fewer than 20 positive source clips. Augmentation will not "
              "fully substitute for more real recordings or synthetic TTS generation.")

    process_class(positive_files, noise_clips, args.out_dir, "positive", N_AUGS_PER_REAL_POSITIVE)
    process_class(negative_files, noise_clips, args.out_dir, "negative", N_AUGS_PER_NEGATIVE)
    process_class(background_files, [], args.out_dir, "background", 1)

    print("Done. Dataset written to", args.out_dir)


if __name__ == "__main__":
    main()
