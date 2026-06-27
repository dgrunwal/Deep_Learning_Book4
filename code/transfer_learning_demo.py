"""
transfer_learning_demo.py  -  Chapter 5 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Two-stage transfer learning on a SMALL local dataset:
  Stage 1 - freeze a pretrained MobileNetV2 base, train a new head.
  Stage 2 - unfreeze the top layers and fine-tune at a LOW learning rate.
Includes basic data augmentation as cheap regularization.

NO DOWNLOAD AT RUN TIME. The images ship with the book in a ./data folder
next to this script:

    data/
        Cat/  000.jpg ... 199.jpg
        Dog/  000.jpg ... 199.jpg

This is a small, hand-picked subset of the Microsoft PetImages set (200 images
per class) -- enough to demonstrate transfer learning, small enough to train in
a couple of minutes on a CPU. If ./data is missing, the script falls back to
synthetic images so it still runs end to end.

Note on hardware: TensorFlow >= 2.11 does not use a GPU on native Windows, so
this runs on the CPU there. The small subset keeps that comfortable. For full
GPU speed, use WSL2 or a cloud notebook (e.g. Google Colab).

Requires: tensorflow   (no tensorflow_datasets, no internet)
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quiet TF logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # consistent CPU math

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (160, 160)
BATCH = 32
EPOCHS_HEAD = 2      # Stage 1 epochs (train the new head)
EPOCHS_FINE = 2      # Stage 2 epochs (fine-tune the top layers)

# Folder of class-subfolders shipped with the book, resolved relative to THIS
# script so it works no matter what directory you run from.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_data():
    """
    Load the local ./data folder with Keras' image_dataset_from_directory,
    which infers the two class labels from the Cat/ and Dog/ subfolder names.
    No download happens. If ./data is absent, fall back to synthetic data.
    """
    if os.path.isdir(DATA_DIR):
        # 80/20 train/validation split from the same folder, fixed seed so the
        # split is reproducible.
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="training", seed=42,
            image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="validation", seed=42,
            image_size=IMG_SIZE, batch_size=BATCH, label_mode="binary")

        # MobileNetV2 expects inputs scaled to [-1, 1]; preprocess_input does that.
        prep = tf.keras.applications.mobilenet_v2.preprocess_input
        train_ds = train_ds.map(lambda x, y: (prep(x), y))
        val_ds = val_ds.map(lambda x, y: (prep(x), y))

        # .cache() keeps the decoded images in memory so epoch 2+ are much faster.
        train_ds = train_ds.cache().shuffle(512).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        print(f"Loaded local images from {DATA_DIR}")
        return train_ds, val_ds

    # Fallback: no data folder present, make tiny synthetic data so the demo runs.
    print(f"(No data folder at {DATA_DIR})")
    print("Falling back to synthetic data so the demo still runs end to end.")
    n = 256
    x = tf.random.normal((n, *IMG_SIZE, 3))
    y = tf.cast(tf.reduce_mean(x, axis=[1, 2, 3]) > 0, tf.float32)[:, None]
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    train_ds = ds.take(200).cache().shuffle(200).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    val_ds = ds.skip(200).cache().batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model():
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # Stage 1: freeze the convolutional base

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)  # binary logit
    return models.Model(inputs, outputs), base


def main():
    train_ds, val_ds = get_data()
    model, base = build_model()

    # Stage 1: train the new head only (the frozen base just extracts features).
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\nStage 1: feature extraction (base frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=2)

    # Stage 2: unfreeze the TOP layers, fine-tune at a LOW learning rate.
    base.trainable = True
    for layer in base.layers[:-30]:        # keep all but the last 30 layers frozen
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # note the tiny LR
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    print("\nStage 2: fine-tuning (top layers unfrozen, LR=1e-5)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, verbose=2)

    print("\nDone. Stage 1 trained a new head on frozen features;")
    print("Stage 2 gently adjusted the top layers. That is transfer learning.")


if __name__ == "__main__":
    main()
