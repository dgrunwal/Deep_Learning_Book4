"""
============================================================================
 deep_learning_facial_recognition.py
 The basics of deep-learning facial recognition: turning a face into a
 128-number "encoding", then comparing encodings by distance.
 © 2026 David Grunwald. All rights reserved.
============================================================================

THE BIG IDEA
------------
A face recognition system does NOT compare photos pixel by pixel - lighting,
angle, and expression change too much for that to work. Instead, a neural
network turns each face image into a short list of numbers called an
ENCODING (here, 128 numbers). The network is trained so that:

    - two pictures of the SAME person  -> encodings that are CLOSE together
    - pictures of DIFFERENT people     -> encodings that are FAR apart

Once you have encodings, everything reduces to measuring the DISTANCE
between two lists of numbers.

TWO RELATED TASKS
-----------------
* VERIFICATION (1:1)  - "Are you who you claim to be?" You compare ONE new
  face against ONE stored face. Easier.
* RECOGNITION (1:K)   - "Who are you?" You compare ONE new face against a
  whole database of K people and pick the closest match (or 'unknown').
  Harder, because there are many more chances to be wrong.

The SAME encoding network powers both tasks - only the comparison logic
differs.

HOW THE NETWORK LEARNS: TRIPLET LOSS
------------------------------------
The encoder is trained with the TRIPLET LOSS. Each training example is a
triplet of three images:
    A = Anchor   (a reference face)
    P = Positive (the SAME person as the anchor)
    N = Negative (a DIFFERENT person)
The loss pushes the anchor-positive distance to be smaller than the
anchor-negative distance by at least a margin (alpha). See triplet_loss()
below. In a real system (like FaceNet) this is how the 128-dim encoding is
learned. This demo uses a tiny stand-in encoder so it runs instantly without
downloading a pretrained model, but the verify/recognize logic is identical
to the real thing.

HOW TO RUN IT
-------------
1. Install the requirements:
       python -m pip install numpy tensorflow
2. Run the script:
       python deep_learning_facial_recognition.py
   It prints the triplet-loss value on a test case, then runs verification
   and recognition demos and explains each result.
============================================================================
"""

import numpy as np
import tensorflow as tf


# ===========================================================================
# STEP 1 - THE TRIPLET LOSS (how a real encoder is trained)
# ---------------------------------------------------------------------------
# This is the exact loss used to TRAIN a face encoder such as FaceNet. We are
# not training here (the demo uses a pretrained/stand-in encoder), but the
# function is included because it is the heart of how the encoding is learned.
#
# For each triplet (Anchor, Positive, Negative):
#   pos_dist = squared distance between anchor and positive encodings
#   neg_dist = squared distance between anchor and negative encodings
#   loss     = max( pos_dist - neg_dist + alpha , 0 )  summed over examples
# The 'alpha' margin forces a real gap, preventing the lazy solution where
# every face maps to the same point.
# ===========================================================================

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract them and add the margin alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: clip negatives to zero, then sum over all the training examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


# ===========================================================================
# STEP 2 - TURN A FACE IMAGE INTO A 128-NUMBER ENCODING
# ---------------------------------------------------------------------------
# In the real assignment this calls a pretrained FaceNet model:
#
#     def img_to_encoding(image_path, model):
#         img = tf.keras.preprocessing.image.load_img(image_path,
#                                                      target_size=(160, 160))
#         img = np.around(np.array(img) / 255.0, decimals=12)
#         x_train = np.expand_dims(img, axis=0)
#         embedding = model.predict_on_batch(x_train)
#         return embedding / np.linalg.norm(embedding, ord=2)
#
# To keep THIS script runnable with no downloads, we simulate that encoder
# with a small fixed random network. It still maps an image to a normalized
# 128-dim vector, which is all the verify / recognize logic needs.
# ===========================================================================

# A fixed "encoder" that stands in for the trained CNN. To behave like a real
# face encoder - same person close, different people far - it summarizes each
# image into coarse regional averages (robust to small noise) and then
# projects to 128 dimensions. This is NOT how real FaceNet works internally;
# it is just enough to make the distances behave sensibly for the demo.
np.random.seed(42)
_PROJ = np.random.randn(8 * 8 * 3, 128)


def img_to_encoding(image_array, model=None):
    """Map a face image (a numpy array) to a normalized 128-dim encoding."""
    img = image_array / 255.0
    # Downsample to an 8x8x3 summary by averaging blocks (robust to noise).
    blocks = img.reshape(8, 20, 8, 20, 3).mean(axis=(1, 3))   # -> (8, 8, 3)
    feat = blocks.reshape(1, -1)
    embedding = feat @ _PROJ                                  # the "network"
    embedding = embedding / np.linalg.norm(embedding, ord=2)  # unit length
    return embedding


# ===========================================================================
# STEP 3 - FACE VERIFICATION (the 1:1 problem)
# ---------------------------------------------------------------------------
# The person swipes an ID card claiming an identity. We encode their live
# photo, look up the stored encoding for that claimed name, and measure the
# distance. Small distance -> same person -> open the door.
#
# The 'threshold' is the cutoff distance. Its value depends on the encoder's
# distance scale: the real FaceNet assignment uses 0.7, while this demo's
# stand-in encoder separates people at much smaller distances, so we use 0.1.
# Either way, the IDEA is identical: below the threshold = same person.
# ===========================================================================

def verify(image_array, identity, database, model=None, threshold=0.1):
    encoding = img_to_encoding(image_array, model)
    dist = np.linalg.norm(encoding - database[identity])   # L2 distance
    if dist < threshold:
        print(f"  It's {identity}, welcome in!   (distance {dist:.4f})")
        door_open = True
    else:
        print(f"  It's not {identity}, please go away.   (distance {dist:.4f})")
        door_open = False
    return dist, door_open


# ===========================================================================
# STEP 4 - FACE RECOGNITION (the 1:K problem)
# ---------------------------------------------------------------------------
# No ID card this time - just a face. We encode it and compare against EVERY
# entry in the database, keeping the closest match. If even the closest is
# farther than the threshold, the person is treated as unknown.
# ===========================================================================

def who_is_it(image_array, database, model=None, threshold=0.1):
    encoding = img_to_encoding(image_array, model)
    min_dist = 100                       # start large; any real distance beats it
    identity = None
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > threshold:
        print(f"  Not in the database.   (closest was {identity} at {min_dist:.4f})")
    else:
        print(f"  It's {identity}, the distance is {min_dist:.4f}")
    return min_dist, identity


# ===========================================================================
# STEP 5 - A RUNNABLE DEMO
# ---------------------------------------------------------------------------
# We fake a few "people" as random images, and create a second, slightly
# noisy photo of one of them to act as a live camera shot. Because our
# stand-in encoder is consistent, the noisy second photo of a person still
# lands close to that person's stored encoding - demonstrating verification
# and recognition end to end.
# ===========================================================================

def make_face(seed):
    """Make a fake 160x160 RGB 'face' with a person-specific look.

    Real faces differ in structure, not just random noise. So we give each
    'person' a distinctive colored gradient + blob pattern, which the encoder
    can pick up - mimicking how a real encoder separates different people.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:160, 0:160] / 160.0
    base = np.zeros((160, 160, 3), dtype='float32')
    for ch in range(3):
        a, b, c = rng.uniform(-1, 1, size=3)
        base[..., ch] = 128 + 110 * (a * xx + b * yy + c * np.sin(6 * xx + seed))
    return np.clip(base, 0, 255).astype('float32')


def noisy(image, seed):
    """A slightly different photo of the SAME face (small added noise)."""
    rng = np.random.RandomState(seed)
    return np.clip(image + rng.normal(0, 5, image.shape), 0, 255).astype('float32')


if __name__ == "__main__":
    print("=" * 64)
    print(" Deep Learning Facial Recognition - demo")
    print("=" * 64)

    # ---- 1. Show the triplet loss on a small test case ----
    tf.random.set_seed(1)
    y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random.normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random.normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(None, y_pred)
    print("\n[Triplet loss] value on a random test triplet:")
    print(f"  loss = {loss.numpy():.6f}")
    print("  (This is the quantity minimized when TRAINING a real encoder.)")

    # ---- 2. Build a small database of known people ----
    print("\n[Database] enrolling three known employees: danielle, younes, kian")
    danielle = make_face(1)
    younes = make_face(2)
    kian = make_face(3)
    database = {
        "danielle": img_to_encoding(danielle),
        "younes":   img_to_encoding(younes),
        "kian":     img_to_encoding(kian),
    }

    # ---- 3. VERIFICATION demo (1:1) ----
    print("\n[Verification - 1:1] 'I am younes' (a fresh photo of younes):")
    camera_younes = noisy(younes, seed=99)
    verify(camera_younes, "younes", database)

    print("\n[Verification - 1:1] 'I am kian' (but it is really a stranger):")
    stranger = make_face(777)
    verify(stranger, "kian", database)

    # ---- 4. RECOGNITION demo (1:K) ----
    print("\n[Recognition - 1:K] who is this? (a fresh photo of younes, no name given):")
    who_is_it(camera_younes, database)

    print("\n[Recognition - 1:K] who is this? (a complete stranger):")
    who_is_it(stranger, database)

    print("\nNotice: the SAME 128-dim encoding powered both verification and")
    print("recognition. Only the comparison logic changed.")
    print("Done.")
