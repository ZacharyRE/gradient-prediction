from __future__ import annotations

import numpy as np


def row_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > eps)
