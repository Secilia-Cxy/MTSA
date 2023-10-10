import numpy as np


class MLForecastModel:

    def __init__(self) -> None:
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """
        :param X: history timesteps
        :param Y: future timesteps to predict
        """
        self._fit(X)
        self.fitted = True

    def _fit(self, X: np.ndarray):
        raise NotImplementedError

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        """
        :param X: history timesteps
        :return: predicted future timesteps
        """
        if not self.fitted:
            raise ValueError("Model has not been trained.")
        pred = self._forecast(X, pred_len)
        return pred
