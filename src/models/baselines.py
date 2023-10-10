import numpy as np

from src.models.base import MLForecastModel


class ZeroForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        return np.zeros((X.shape[0], pred_len))


class MeanForecast(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        mean = np.mean(np.mean(X, axis=-1), axis=0)
        return mean * np.ones((X.shape[0], pred_len))

# TODO: add other models based on MLForecastModel
