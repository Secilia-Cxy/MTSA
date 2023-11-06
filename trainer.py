import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from src.utils.metrics import mse, mae


class MLTrainer:
    def __init__(self, model, transform, dataset):
        self.model = model
        self.transform = transform
        self.dataset = dataset

    def train(self):
        train_X = self.dataset.train_data
        t_X = self.transform.transform(train_X)
        self.model.fit(t_X)

    def evaluate(self, dataset, seq_len=96, pred_len=32):
        if dataset.type == 'm4':
            test_X = dataset.train_data
            test_Y = dataset.test_data
            pred_len = dataset.test_data.shape[-1]
        else:
            test_data = dataset.test_data
            subseries = np.concatenate(([sliding_window_view(v, (seq_len + pred_len, v.shape[-1])) for v in test_data]))
            test_X = subseries[:, 0, :seq_len, :]
            test_Y = subseries[:, 0, seq_len:, :]
        te_X = self.transform.transform(test_X)
        fore = self.model.forecast(te_X, pred_len=pred_len)
        fore = self.transform.inverse_transform(fore)
        print('mse:', mse(fore, test_Y))
        print('mae:', mae(fore, test_Y))
