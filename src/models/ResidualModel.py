import numpy as np

from src.models.base import MLForecastModel


class ResidualModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        raise NotImplementedError
