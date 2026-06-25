"""
style_loss_demo.py  -  Chapter 9 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Builds a Gram matrix from a feature map and computes the style loss between two
images. The Gram matrix captures correlations between feature channels, which
is what encodes "style" (texture) independent of content (layout).
"""
import numpy as np


def gram_matrix(features):
    """
    features: array of shape (channels, height*width)
    Returns the (channels x channels) Gram matrix of channel correlations,
    normalized by the number of spatial positions.
    """
    c, n = features.shape
    return (features @ features.T) / n


def style_loss(feat_a, feat_b):
    """Mean squared error between the two Gram matrices."""
    g_a = gram_matrix(feat_a)
    g_b = gram_matrix(feat_b)
    return np.mean((g_a - g_b) ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Pretend these are flattened activations from one CNN layer:
    # 3 channels, 16 spatial positions (a 4x4 feature map).
    style_image   = rng.normal(size=(3, 16))
    target_same   = style_image + rng.normal(scale=0.01, size=(3, 16))  # similar
    target_diff   = rng.normal(size=(3, 16))                            # different

    print("Gram matrix of the style image:\n", np.round(gram_matrix(style_image), 3))
    print(f"\nStyle loss vs. a SIMILAR feature map:  {style_loss(style_image, target_same):.5f}")
    print(f"Style loss vs. a DIFFERENT feature map: {style_loss(style_image, target_diff):.5f}")

    print("\nNeural style transfer minimizes a weighted sum of this style loss")
    print("and a content loss, optimizing the IMAGE rather than the network.")
