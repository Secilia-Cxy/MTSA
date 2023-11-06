import numpy as np


def euclidean(A, B):
    return np.sqrt(np.sum(np.sum((A - B) ** 2, axis=1), axis=1))

# TODO: add more distance

