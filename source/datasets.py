import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split


class HydroDataset(Dataset):
    # load the dataset
    def __init__(self, path, seq_length, input_size, output_size):
        # load the csv file as a dataframe
        if isinstance(path, pd.DataFrame):
            df = path
        else:
            df = pd.read_csv(path)
        # store the inputs and outputs
        X = df.values[:, :input_size].astype('float32')
        y = df.values[:, input_size:input_size +
                      output_size].astype('float32')[seq_length - 1:]
        # seq_length inputs to one target
        y_len = y.shape[0]
        idx_add = np.tile(np.arange(0, seq_length), y_len)
        idx_rep = np.repeat(np.arange(0, y_len), seq_length)
        X = X[idx_add + idx_rep]
        # ensure inputs and target has the right shape
        self.X = X.reshape(y_len, seq_length, input_size)
        self.y = y.reshape((y_len, output_size))
        # initial normalization
        self.norm = None
        self.scaler()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

    # normalization
    def scaler(self, norm=None):
        if self.norm is not None:
            return
        # scaler of X
        if norm is None:
            X_mean = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0)
            X_std = self.X.reshape(-1, self.X.shape[2]).std(axis=0)
        else:
            X_mean = norm[0]
            X_std = norm[1]
        # scaler of y
        if norm is None:
            y_mean = 0
            y_std = self.y.max(axis=0) - y_mean
        else:
            y_mean = norm[2]
            y_std = norm[3]
        # normalize X and y
        self.y = (self.y - y_mean) / y_std
        self.X = (self.X - X_mean) / X_std
        self.norm = (X_mean, X_std, y_mean, y_std)
        return
