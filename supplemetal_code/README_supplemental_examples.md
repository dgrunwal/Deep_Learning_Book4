# Supplemental In-Line Code Examples
## Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
*A Practice-First Path from Pixels to Production*
© 2026 David Grunwald. All rights reserved.

Repository: https://github.com/dgrunwal/Deep_Learning_Book4

These four worked examples accompany Book 4. Where the ten short companion
scripts each isolate a single idea, these four show that idea assembled into a
complete, runnable program you can read, execute, and modify. They are the
in-line examples referenced from the chapters.

Each has been cleaned up from its original notebook form: course-specific
grader scaffolding and helper imports have been removed, every function is
commented for a beginner, and each script runs standalone with no hidden
dependencies. The two TensorFlow examples fall back to small synthetic data
when no real dataset is supplied, so they run anywhere while still showing the
full training loop.

## The four examples

| Script | Chapters | What it demonstrates | Stack |
|--------|----------|----------------------|-------|
| `ex1_conv_from_scratch.py` | 1–2 | Padding, single-step convolution, full convolution, and pooling — built by hand. | NumPy |
| `ex2_cnn_in_keras.py` | 3 | The same CNN built two ways: the Sequential and Functional Keras APIs. | TensorFlow |
| `ex3_resnet_blocks.py` | 4 | ResNet's identity and convolutional blocks, and why skip connections matter. | TensorFlow |
| `ex4_mobilenet_transfer.py` | 5 | Two-stage transfer learning on MobileNetV2: feature extraction, then fine-tuning. | TensorFlow |

 

## Requirements

- Python 3.10+ (tested under 3.13)
- `numpy` (all examples)
- `tensorflow` (examples 2, 3, and 4)

Install:

    pip install numpy tensorflow

## Running

Each example runs standalone:

    python ex1_conv_from_scratch.py
    python ex2_cnn_in_keras.py
    python ex3_resnet_blocks.py
    python ex4_mobilenet_transfer.py

Notes:

- **Example 1** needs only NumPy and prints output shapes you can check against
  the Chapter 2 size formula `out = (in - f + 2*pad) / stride + 1`.
- **Examples 2 and 3** train and run a forward pass on small synthetic images,
  so they need no downloads.
- **Example 4** downloads the pretrained MobileNetV2 weights (about 14 MB) the
  first time it builds the model. To train on real images, edit the `DATA_DIR`
  line near the top of the file to point at a folder of class-subfolders:

      my_data/
          class_a/   (.jpg files)
          class_b/   (.jpg files)

  If `DATA_DIR` is not found, the script uses synthetic data so it still runs
  end to end.

## House conventions

- Beginner-readable, heavily commented.
- NumPy-first; frameworks only where they earn their place.
- Aligned with the data and training conventions of Book 1 (ML For Beginners)
  and Book 2 (Improving Deep Neural Networks).
