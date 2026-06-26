"""
============================================================================
 object_detection_self_driving.py
 The basics of how a self-driving car "sees" the road, using a U-Net for
 SEMANTIC IMAGE SEGMENTATION (labeling every pixel of the scene).
 © 2026 David Grunwald. All rights reserved.
============================================================================

THE BIG IDEA
------------
A self-driving car has to understand a camera image at the level of every
single pixel: which pixels are road, which are cars, pedestrians, lane lines,
sky, and so on. That pixel-by-pixel labeling is called SEMANTIC IMAGE
SEGMENTATION.

This script builds a U-Net, the classic neural network for this job. A U-Net
has two halves that give it its "U" shape:

    ENCODER (downsampling)            DECODER (upsampling)
    shrinks the image, learning  -->  grows it back to full size, so the
    *what* is in the scene            output mask is the same resolution
                                      as the input image

Crucially, the encoder and decoder are connected by SKIP CONNECTIONS that
carry fine spatial detail straight across the "U", so the final mask has
crisp, correctly-placed borders.

THREE THINGS A BEGINNER SHOULD REMEMBER
---------------------------------------
1. Semantic image segmentation predicts a label for EVERY SINGLE PIXEL in an
   image (not just one label for the whole picture).
2. U-Net uses an EQUAL NUMBER of convolutional blocks (downsampling) and
   transposed convolutions (upsampling) - that symmetry is the "U".
3. SKIP CONNECTIONS pass encoder feature maps directly to the matching
   decoder stage. They prevent loss of border-pixel information and help
   reduce overfitting.

HOW TO RUN IT
-------------
1. Install the requirements:
       python -m pip install tensorflow numpy
2. Run the script:
       python object_detection_self_driving.py
   It builds the U-Net and prints a model summary. No dataset is required -
   the goal here is to SEE the architecture, not to train for hours.
============================================================================
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,
                                      Conv2DTranspose, concatenate)


# ===========================================================================
# STEP 1 - PREPROCESS ONE IMAGE / MASK PAIR
# ---------------------------------------------------------------------------
# Before the network sees anything, each camera image is:
#   - read from disk and decoded into pixels,
#   - resized to a fixed size the network expects,
#   - normalized so pixel values fall in [0, 1] (dividing by 255).
# The "mask" is the answer key: an image where each pixel's value IS the
# class label for that pixel (0 = road, 1 = car, 2 = pedestrian, ...).
# We do NOT normalize the mask, because its numbers are labels, not bright-
# ness values.
# ===========================================================================

def process_path(image_path, mask_path):
    """Read and decode an image and its segmentation mask."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)   # scales to [0, 1]

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    # Collapse the 3 identical channels down to a single label channel
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask


def preprocess(image, mask):
    """Resize both image and mask to the network's input size."""
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')
    return input_image, input_mask


# ===========================================================================
# STEP 2 - THE ENCODER BLOCK (downsampling / "contracting" path)
# ---------------------------------------------------------------------------
# Each encoder block does:
#   - two Conv2D layers that detect features (edges, textures, shapes),
#   - an optional Dropout layer (randomly switches off some neurons during
#     training to fight OVERFITTING),
#   - an optional MaxPooling2D layer that halves the width and height,
#     so deeper layers see a larger area of the scene at once.
#
# It returns TWO things:
#   next_layer       -> goes deeper into the network
#   skip_connection  -> a copy taken BEFORE pooling, saved to hand across the
#                       "U" to the decoder later. This is the skip connection.
# ===========================================================================

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    # Two convolutions. 'same' padding keeps the size; 'he_normal' is a good
    # weight initializer for ReLU networks.
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)

    # Dropout only in the deeper blocks, to reduce overfitting.
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    # Pool to shrink the spatial size (except in the bottleneck).
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv          # saved for the matching decoder stage
    return next_layer, skip_connection


# ===========================================================================
# STEP 3 - THE DECODER BLOCK (upsampling / "expanding" path)
# ---------------------------------------------------------------------------
# Each decoder block does the reverse of an encoder block:
#   - Conv2DTranspose UPSAMPLES (doubles width and height),
#   - it then CONCATENATES the matching encoder skip connection. This is the
#     step that restores precise border detail the pooling threw away,
#   - two more Conv2D layers refine the merged result.
#
# Because there is exactly one decoder block per encoder block, the network
# is symmetric - the defining property of a U-Net.
# ===========================================================================

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    # Upsample the deeper features back up toward full resolution.
    up = Conv2DTranspose(n_filters, 3, strides=(2, 2),
                         padding='same')(expansive_input)

    # SKIP CONNECTION: glue the encoder's saved features onto the upsampled
    # ones. Order [up, contractive_input] matters for shape alignment.
    merge = concatenate([up, contractive_input], axis=3)

    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(conv)
    return conv


# ===========================================================================
# STEP 4 - ASSEMBLE THE FULL U-NET
# ---------------------------------------------------------------------------
# Encoder: 5 conv_blocks, DOUBLING the filter count each step (32, 64, 128,
#          256, 512). The last block is the "bottleneck": dropout on, pooling
#          off. The deeper blocks add dropout to combat overfitting.
# Decoder: 4 upsampling_blocks, HALVING the filters each step and pulling in
#          the matching encoder skip connection (note: we use the SECOND
#          output of each encoder block - the copy taken before pooling).
# Output : a Conv2D with n_classes filters and a 1x1 kernel. This produces,
#          for every pixel, a score for each possible class. The class with
#          the highest score becomes that pixel's predicted label.
# ===========================================================================

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)

    # ---- Encoder (downsampling) ----
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3,
                         max_pooling=False)          # bottleneck

    # ---- Decoder (upsampling) ----  uses skip connections cblockN[1]
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(ublock9)

    # One score per class, per pixel.
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model


# ===========================================================================
# STEP 5 - TURN MODEL OUTPUT INTO A PREDICTED MASK
# ---------------------------------------------------------------------------
# The model outputs n_classes scores per pixel. argmax over that last axis
# picks the winning class for each pixel, producing the final segmentation
# mask the car would actually use.
# ===========================================================================

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)   # winning class per pixel
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


# ===========================================================================
# STEP 6 - BUILD IT AND LOOK AT IT
# ---------------------------------------------------------------------------
# Running this file constructs the U-Net and prints its summary so you can
# see the downsampling path, the bottleneck, and the upsampling path mirror
# each other. To actually TRAIN it you would compile with an Adam optimizer
# and SparseCategoricalCrossentropy loss, then call model.fit() on a labeled
# driving dataset - that part is left out so this stays fast and dependency-
# light.
# ===========================================================================

if __name__ == "__main__":
    print("=" * 64)
    print(" Object Detection for Self-Driving Cars - U-Net architecture")
    print("=" * 64)

    img_height, img_width, num_channels = 96, 128, 3
    n_classes = 23      # e.g. road, car, pedestrian, lane line, sky, ...

    unet = unet_model((img_height, img_width, num_channels),
                      n_filters=32, n_classes=n_classes)

    print(f"\nInput image size : {img_height} x {img_width} x {num_channels}")
    print(f"Number of classes: {n_classes} (one label per pixel)\n")
    unet.summary()

    # How you WOULD compile it for training:
    unet.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    # Sanity check: push one random "image" through to confirm output shape.
    dummy = np.random.rand(1, img_height, img_width, num_channels).astype('float32')
    out = unet.predict(dummy, verbose=0)
    print(f"\nRaw output shape : {out.shape}  (1, H, W, n_classes)")
    mask = create_mask(out)
    print(f"Predicted mask   : {mask.shape}  (one class label per pixel)")
    print("\nThe encoder and decoder are symmetric, joined by skip connections.")
    print("Done.")
