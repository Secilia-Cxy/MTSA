import numpy as np


def euclidean(A, B):
    return np.linalg.norm(B - A, axis=1)

# TODO: add more distance
