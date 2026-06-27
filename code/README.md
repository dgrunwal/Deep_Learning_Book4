# Companion Python Scripts
## Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Ten scripts, one per chapter. Eight use NumPy only so the mechanics stay
visible; two use TensorFlow/Keras where a beginner would reach for them in
real work. Each script is short enough to read in one sitting and prints a
small, self-explanatory demo when run directly.

## Chapter placement

| Chapter | Script                      | Stack              |
|---------|-----------------------------|--------------------|
| 1       | conv_from_scratch.py        | NumPy only         |
| 2       | shape_calculator.py         | NumPy only         |
| 3       | cifar10_baseline.py         | TensorFlow / Keras |
| 4       | residual_block.py           | NumPy only         |
| 5       | transfer_learning_demo.py*  | TensorFlow / Keras | *--> data.zip unzip into /data folder
| 6       | iou_demo.py                 | NumPy only         |
| 7       | transpose_conv_demo.py      | NumPy only         |
| 8       | triplet_loss_demo.py        | NumPy only         |
| 9       | style_loss_demo.py          | NumPy only         |
| 10      | conv1d_3d_demo.py           | NumPy only         |

## Requirements

- Python 3.10+ (tested under 3.13)
- numpy                       (all scripts)
- tensorflow                  (cifar10_baseline.py, transfer_learning_demo.py)
- tensorflow_datasets         (transfer_learning_demo.py, optional)

Install:

    pip install numpy tensorflow tensorflow_datasets

## Running

Each script runs standalone:

    python conv_from_scratch.py
    python shape_calculator.py
    ...

The eight NumPy scripts have no external data dependencies. The two
TensorFlow scripts download standard datasets (CIFAR-10, cats_vs_dogs)
on first run.

For script transfer_learning_demo.py in chapter 5 read the short readme DATA_README_CH_5.  You create a folder
/data , unzip and place the data.zip contents consisting of a folder called Cat and Dog in the /data folder where the 
script is run.

## House conventions

- Beginner-readable, heavily commented.
- NumPy-first; frameworks only where they earn their place.
- Aligned with the data conventions of Book 1 (ML For Beginners) and the
  optimization work in Book 2 (Improving Deep Neural Networks).
