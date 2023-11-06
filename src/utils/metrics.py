import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


def mae(predict, target):
    return np.mean(np.abs(predict - target))
