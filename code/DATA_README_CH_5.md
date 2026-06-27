# Chapter 5 Image Data — Cats vs. Dogs

**Convolutional Neural Networks for Beginners** (AI Foundations Series, Book 4)
*A Practice-First Path from Pixels to Production*
© 2026 David Grunwald. All rights reserved.

This folder holds the small image dataset used by `transfer_learning_demo.py`
in Chapter 5. It lets the transfer-learning demo run in a couple of minutes on a
CPU with **no download at run time**.

---

## Setup — unzip the data before running

The images ship as **`data.zip`**. Before running the demo, unzip it so that a
folder named **`data`** sits next to `transfer_learning_demo.py`, with two
class-subfolders inside it:

```
your-folder/
    transfer_learning_demo.py
    data/
        Cat/   000.jpg ... 199.jpg
        Dog/   000.jpg ... 199.jpg
```

### Windows

1. Place `data.zip` in the same folder as `transfer_learning_demo.py`.
2. Right-click `data.zip` and choose **Extract All...** (or use 7-Zip → *Extract Here*).
3. Confirm you end up with a `data` folder containing `Cat` and `Dog` subfolders —
   **not** a nested `data\data\Cat`. If you get an extra nesting level, move the
   inner `data` folder up one level.

### macOS / Linux

```
unzip data.zip
```

Run it from the folder that contains `transfer_learning_demo.py`.

### Verify the layout

From the folder containing the script, this should report `{'Cat': 200, 'Dog': 200}`:

```
python -c "import os; p='data'; print({d: len(os.listdir(os.path.join(p,d))) for d in os.listdir(p)})"
```

---

## Running the demo

Once `data/` is in place:

```
python transfer_learning_demo.py
```

To use a differently named or located folder of two class-subfolders, pass `--data`:

```
python transfer_learning_demo.py --data my_images
```

If no data folder is found, the script falls back to synthetic images so it
still runs end to end (you just won't see meaningful accuracy).

---

## About the images

These 400 images (200 cats, 200 dogs) are a small subset of the **Microsoft
Kaggle Cats and Dogs Dataset** (the "PetImages" collection), provided here for
**educational use only** in support of this book's transfer-learning chapter.

- Source: Microsoft Download Center — *Kaggle Cats and Dogs Dataset*.
- The subset was selected and re-saved as clean JPEGs for teaching purposes; a
  few known-corrupt files in the original distribution were excluded.
- All rights to the underlying images remain with their respective owners. The
  dataset is used here under Microsoft's research/educational terms. No
  ownership of the images is claimed.
- If you wish to use the full dataset (~25,000 images) or use these images
  beyond following along with this book, please obtain them directly from
  Microsoft and review the accompanying terms.

To rebuild or resize the subset from the full PetImages download yourself, use
the `make_pet_subset.py` utility:

```
python make_pet_subset.py --src "path/to/PetImages" --out data --per-class 200
```

---

*Note on hardware:* TensorFlow ≥ 2.11 does not use a GPU on native Windows, so
the demo runs on the CPU there. The small subset keeps that fast. For full GPU
speed, use WSL2 or a cloud notebook (e.g. Google Colab).
