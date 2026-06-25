"""
triplet_loss_demo.py  -  Chapter 8 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Computes the triplet loss on toy anchor / positive / negative embeddings and
shows how changing the margin changes the loss. This is the objective behind
face-recognition embeddings and Siamese networks.

  loss = max(0,  d(anchor, positive) - d(anchor, negative) + margin)
"""
import numpy as np


def dist(a, b):
    """Squared Euclidean distance between two embedding vectors."""
    return np.sum((a - b) ** 2)


def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_d = dist(anchor, positive)   # want this SMALL
    neg_d = dist(anchor, negative)   # want this LARGE
    return max(0.0, pos_d - neg_d + margin), pos_d, neg_d


if __name__ == "__main__":
    # Three 4-D embeddings. Positive is close to the anchor; negative is far.
    anchor   = np.array([0.10, 0.20, 0.30, 0.40])
    positive = np.array([0.12, 0.19, 0.33, 0.38])  # same identity
    negative = np.array([0.80, 0.10, 0.05, 0.60])  # different identity

    print("d(anchor, positive) should be SMALL")
    print("d(anchor, negative) should be LARGE\n")

    for margin in (0.0, 0.2, 0.5, 1.0):
        loss, pd, nd = triplet_loss(anchor, positive, negative, margin)
        print(f"margin={margin:<4}  pos_d={pd:.3f}  neg_d={nd:.3f}  loss={loss:.3f}")

    print("\nWhen the negative is already far enough (by the margin), loss = 0")
    print("and no gradient flows. The margin sets how hard 'far enough' is.")
