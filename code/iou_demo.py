"""
iou_demo.py  -  Chapter 6 companion
Convolutional Neural Networks for Beginners (AI Foundations Series, Book 4)
(c) 2026 David Grunwald. All rights reserved.

Computes Intersection over Union (IoU) for two bounding boxes and draws an
ASCII overlap map so the core detection metric stops being abstract.
Boxes are given as (x1, y1, x2, y2) with (x1, y1) top-left, (x2, y2) bottom-right.
"""


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # intersection rectangle
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def ascii_map(box_a, box_b, grid=12):
    """Crude visualization: A, B, or X where they overlap."""
    rows = []
    for y in range(grid):
        line = []
        for x in range(grid):
            in_a = box_a[0] <= x < box_a[2] and box_a[1] <= y < box_a[3]
            in_b = box_b[0] <= x < box_b[2] and box_b[1] <= y < box_b[3]
            line.append("X" if in_a and in_b else "A" if in_a else "B" if in_b else ".")
        rows.append(" ".join(line))
    return "\n".join(rows)


if __name__ == "__main__":
    box_a = (2, 2, 8, 8)
    box_b = (5, 4, 11, 10)
    print(f"Box A: {box_a}")
    print(f"Box B: {box_b}")
    print(f"\nIoU = {iou(box_a, box_b):.3f}\n")
    print("Overlap map (A=box A, B=box B, X=both):\n")
    print(ascii_map(box_a, box_b))
    print("\nA detection counts as correct when IoU exceeds a threshold,")
    print("commonly 0.5. This single number drives the whole evaluation.")
