import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.models.base import MLForecastModel
from src.utils.tools import EarlyStopping, adjust_learning_rate

class DLDataset(Dataset):
    def __init__(self, X, seq_len, pred_len, mode='train'):
        if mode == 'predict':
            self.data = X
        else:
            self.data = X[0]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode

    def __len__(self):
        if self.mode == 'predict':
            return self.data.shape[0]
        else:
            return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        if self.mode == 'predict':
            x = self.data[idx]
            return x
        else:
            x = self.data[idx: idx + self.seq_len]
            y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
            return x, y


class DLForecastModel(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.optimizer = None
        self.model = None
        self.args = args
        if self.args.device == 'cpu':
            self.device = 'cpu'
        else:
            self.device = f'cuda:{self.args.device}'
        self.criterion = nn.MSELoss()

    def _fit(self, train_X: np.ndarray, val_X=None):
        train_loader = DataLoader(DLDataset(train_X, self.args.seq_len, self.args.pred_len),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(DLDataset(val_X, self.args.seq_len, self.args.pred_len),
                                batch_size=self.args.batch_size,
                                shuffle=False)

        path = os.path.join('checkpoints', f'{self.args.model}_{self.args.dataset}_{self.args.pred_len}')
        if not os.path.exists(path):
            os.makedirs(path)

        # train
        self.model.train()
        train_epochs = self.args.epochs
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(train_epochs):
            train_loss = 0
            epoch_time = time.time()
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss = train_loss / len(train_loader)

            # validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :]
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print("Epoch: {} cost time: {} Train Loss: {} Val Loss: {}".format(epoch + 1, time.time() - epoch_time,
                                                                               train_loss, val_loss))
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        test_loader = DataLoader(DLDataset(X, self.args.seq_len, self.args.pred_len, mode='predict'),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)
        # predict
        self.model.eval()
        fore = []
        with torch.no_grad():
            for batch_idx, (batch_x) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                fore.append(outputs.cpu().numpy())
        fore = np.concatenate(fore, axis=0)
        return fore
