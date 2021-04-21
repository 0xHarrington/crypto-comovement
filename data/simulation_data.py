#!/usr/bin/env python

# Standard imports
import pickle
import os.path
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch.utils.data import DataLoader, Dataset

# Local files
from utils.coin_helpers import load_coins, simplify


"""simulation_data.py: Wrapper for pytorch dataset and dataloader making portfolio simulations easier."""

class SimulationDataset():

    class CryptoReturnsDataset(Dataset):
        def __init__(self, ts: pd.core.frame.DataFrame, lag: int):
            self.raw = ts
            self.lag = lag
            self.n_returns, self.n_coins = self.raw.shape
            self.n_samples = self.n_returns - self.lag

            self.samples = np.zeros((self.n_samples, self.lag, self.n_coins))
            for i, samp in enumerate(self.raw.rolling(window=lag)):
                ind = i - lag + 1
                if ind >= 0 and ind < self.n_samples:
                    self.samples[ind] = samp.values

        def __getitem__(self, index):
            x = torch.tensor(self.samples[index])
            y = torch.tensor(self.raw.iloc[index + 1].values)
            return x.float(), y.float()

        def __len__(self):
            return self.n_samples

    # ================================================================

    def __init__(self, subset: list, interval = '1D', lag = 1):
        """Create the CryptoReturnsDataset and prepare the instance variables"""

        filename = f"{len(subset)}-coins_{interval}-returns.pkl"
        dir_f = f'data/{filename}'

        # Check for previously-loaded data
        if os.path.isfile(dir_f):
            f = open(dir_f, 'rb')
            self.full_raw = pickle.load(f)
            f.close()
            print(f'{filename} found and loaded!')
        else:
            print(f'Couldn\'t find pickled raw data for {interval}-{len(subset)} coins')
            # Load all the data
            _, ts = load_coins('data/pairs/', subset)
            ts = ts.resample(interval, label='right', closed='right', axis=0).asfreq()
            self.full_raw = ts.dropna(0, 'any')

            # Pickle and save for next time
            f = open(dir_f, 'wb')
            pickle.dump(self.full_raw, f)
            f.close()
            print(f'{filename} saved in `data` directory!')

        self.lag = lag
        self.n_coins = len(self.full_raw.columns)
        self.n_returns = self.full_raw.shape[0]
        self.train_test_thresh = round(self.n_returns * .8) # hard-coded 80/20 train/test
        self.n_oos = self.n_returns - self.train_test_thresh

    def out_of_sample_shape(self):
        """Returns the number of out-of-sample testing steps
        :returns: int
        """
        return self.n_oos - self.lag, self.n_coins

    def in_sample_shape(self):
        """Returns the number of in-sample training samples
        :returns: int
        """
        return self.train_test_thresh - self.lag, self.n_coins

    def get_training(self, index: int):
        """Get a training data-sized training set, starting from index. Currently hard-coded to 80% of the input time series.
        :index: Number of first sample in the returned DataLoader
        :returns: PyTorch DataLoader
        """

        #  verify index is in bounds
        if index > self.train_test_thresh:
            raise IndexError("Simulation Dataset was passed a training set index beyond the train/test split")

        ds = self._get_subset(index, index + self.train_test_thresh + self.lag - 1)
        return DataLoader(ds, batch_size=64, shuffle=True)

    def get_out_of_sample(self):
        """ Get the testing data from the input dataset. Currently hard-coded to the later 20% of the input timeseries
        :returns: PyTorch DataLoader covering the testing samples
        """
        ds = self._get_subset(self.train_test_thresh, self.n_returns)
        return DataLoader(ds, batch_size=1, shuffle=True)

    def _get_subset(self, start, end):
        """Helper method for indexing into stored dataset
        :start: Beginning of subset
        :end: End of subset
        :returns: Pandas DataFrame of coin returns
        """
        return self.CryptoReturnsDataset(self.full_raw.iloc[start : end], self.lag)
