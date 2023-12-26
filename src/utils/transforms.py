import numpy as np


class Transform:
    """
    Preprocess time series
    """

    def transform(self, data):
        """
        :param data: raw timeseries
        :return: transformed timeseries
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: raw timeseries
        :return: inverse_transformed timeseries
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    def __init__(self, args):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class StandardizationTransform(Transform):
    def __init__(self, args):
        super().__init__()
        self.fitted = False
        self.eps = 1e-5

    def transform(self, data):
        if not self.fitted:
            self.means = np.mean(data, axis=1)
            self.stds = np.std(data, axis=1)
            self.fitted = True
        trans_data = (data - self.means) / (self.stds + self.eps)  # 防止分母为0

        return trans_data

    def inverse_transform(self, data):
        inv_trans_data = data * (self.stds + self.eps) + self.means

        return inv_trans_data
