import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.models.base import MLForecastModel
from src.utils.distance import euclidean


class TsfKNN(MLForecastModel):
    def __init__(self, args):
        self.k = args.n_neighbors
        if args.distance == 'euclidean':
            self.distance = euclidean
        self.msas = args.msas
        super().__init__()

    def _fit(self, X: np.ndarray) -> None:
        self.X = X[0, :, -1]

    def _search(self, x, X_s, seq_len, pred_len):
        if self.msas == 'MIMO':
            distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len:]
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            return x_fore
        elif self.msas == 'recursive':
            distances = self.distance(x, X_s[:, :seq_len])
            indices_of_smallest_k = np.argsort(distances)[:self.k]
            neighbor_fore = X_s[indices_of_smallest_k, seq_len].reshape((-1, 1))
            x_fore = np.mean(neighbor_fore, axis=0, keepdims=True)
            x_new = np.concatenate((x[:, 1:], x_fore), axis=1)
            if pred_len == 1:
                return x_fore
            else:
                return np.concatenate((x_fore, self._search(x_new, X_s, seq_len, pred_len - 1)), axis=1)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        fore = []
        seq_len = X.shape[1]
        X_s = sliding_window_view(self.X, seq_len + pred_len)
        for x in X:
            x = np.expand_dims(x, axis=0)
            x_fore = self._search(x, X_s, seq_len, pred_len)
            fore.append(x_fore)
        fore = np.concatenate(fore, axis=0)
        return fore
