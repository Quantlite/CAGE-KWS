"""
Trains DS-CNN-Small (depthwise-separable CNN) on log-mel features for
3-class keyword spotting: positive ("hey astra") / negative (other speech)
/ background (noise, silence).

3 classes instead of binary lets the model distinguish "confidently not
speech" from "speech but not the keyword" -- both matter for the false
activation control logic downstream (see PS requirement on near-zero
false activations).

Usage:
    python train.py --data_dir dataset --epochs 40 --out_dir model_out
"""
import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from features import log_mel, feature_shape

CLASSES = ["background", "negative", "positive"]  # index order matters


def load_dataset_split(data_dir, split):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        files = glob.glob(os.path.join(data_dir, split, cls, "*.wav"))
        paths.extend(files)
        labels.extend([idx] * len(files))
    return paths, labels


def make_feature_extractor():
    import soundfile as sf

    def _extract(path_tensor):
        path = path_tensor.numpy().decode("utf-8")
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        feat = log_mel(y.astype(np.float32))
        return feat.astype(np.float32)

    return _extract


def spec_augment(feat, freq_mask=6, time_mask=10):
    """Lightweight SpecAugment: mask random frequency/time bands to reduce
    overfitting to specific recording conditions."""
    feat = tf.identity(feat)
    n_frames = tf.shape(feat)[0]
    n_mels = tf.shape(feat)[1]

    f0 = tf.random.uniform([], 0, tf.maximum(n_mels - freq_mask, 1), dtype=tf.int32)
    freq_indices = tf.range(n_mels)
    freq_mask_bool = (freq_indices >= f0) & (freq_indices < f0 + freq_mask)
    feat = tf.where(freq_mask_bool[tf.newaxis, :], tf.zeros_like(feat), feat)

    t0 = tf.random.uniform([], 0, tf.maximum(n_frames - time_mask, 1), dtype=tf.int32)
    time_indices = tf.range(n_frames)
    time_mask_bool = (time_indices >= t0) & (time_indices < t0 + time_mask)
    feat = tf.where(time_mask_bool[:, tf.newaxis], tf.zeros_like(feat), feat)
    return feat


def build_dataset(paths, labels, batch_size, training):
    extractor = make_feature_extractor()
    shape = feature_shape()

    def gen():
        idxs = np.arange(len(paths))
        if training:
            np.random.shuffle(idxs)
        for i in idxs:
            yield paths[i], labels[i]

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    def _map(path, label):
        feat = tf.py_function(extractor, [path], tf.float32)
        feat.set_shape(shape)
        if training:
            feat = spec_augment(feat)
        feat = tf.expand_dims(feat, -1)  # (frames, mels, 1)
        return feat, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def ds_cnn_small(input_shape, n_classes):
    """DS-CNN-Small: ~ a few hundred K MACs, ~24-30K params depending on
    channel width. Depthwise-separable convs give the compute/RAM headroom
    needed for <256KB / <10% CPU on ESP32-S3."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for filters in (64, 64, 64):
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="ds_cnn_small")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="model_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_paths, train_labels = load_dataset_split(args.data_dir, "train")
    val_paths, val_labels = load_dataset_split(args.data_dir, "val")
    print(f"train: {len(train_paths)}  val: {len(val_paths)}")

    # Class weights: positives are almost always the minority class.
    counts = np.bincount(train_labels, minlength=len(CLASSES))
    total = counts.sum()
    class_weight = {i: total / (len(CLASSES) * max(c, 1)) for i, c in enumerate(counts)}
    print("class counts:", dict(zip(CLASSES, counts)), " class_weight:", class_weight)

    train_ds = build_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_ds = build_dataset(val_paths, val_labels, args.batch_size, training=False)

    shape = feature_shape() + (1,)
    model = ds_cnn_small(shape, len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, "best.weights.h5"),
            save_best_only=True, save_weights_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weight, callbacks=callbacks,
    )

    # Keras 3 (TF 2.16+) no longer supports directory-style SavedModel via
    # model.save() -- that now requires a .keras or .h5 file extension.
    # model.export() is the Keras 3 way to produce a TF SavedModel directory,
    # which is what quantize.py's TFLiteConverter.from_saved_model() expects.
    model.export(os.path.join(args.out_dir, "kws_model"))
    # Also keep a native Keras checkpoint for easy reloading/fine-tuning later.
    model.save(os.path.join(args.out_dir, "kws_model.keras"))
    print("Saved SavedModel to", os.path.join(args.out_dir, "kws_model"))


if __name__ == "__main__":
    main()
