"""Box geometry. Boxes are [x1, y1, x2, y2] in pixels."""


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def overlap_frac(inner, outer) -> float:
    """Fraction of `inner` inside `outer`. Not IoU: small glasses fully inside a big bag are 1.0."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def base_point(b) -> tuple[float, float]:
    """Bottom-centre: where an object touches the surface it sits on."""
    return (b[0] + b[2]) / 2.0, b[3]


def contains(box, pt, margin=0.0) -> bool:
    return (box[0] - margin <= pt[0] <= box[2] + margin
            and box[1] - margin <= pt[1] <= box[3] + margin)
