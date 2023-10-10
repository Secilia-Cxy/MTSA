import numpy as np


def mse(predict, target):
    return np.mean((target - predict) ** 2)


# TODO: implement the metrics
def mae(predict, target):
    raise NotImplementedError


def mape(predict, target):
    raise NotImplementedError


def smape(predict, target):
    raise NotImplementedError


def mase(predict, target):
    raise NotImplementedError
