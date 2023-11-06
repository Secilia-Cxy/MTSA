import torch.nn as nn
import numpy as np

from src.models.base import MLForecastModel


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = Model(args)

    def _fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        raise NotImplementedError


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = individual
        self.channels = configs.enc_in

        # TODO: implement the following layers

    def forward(self, x):
        raise NotImplementedError

        # TODO: implement the forward pass
