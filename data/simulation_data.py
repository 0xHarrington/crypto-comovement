#!/usr/bin/env python

# Local files
from utils.coin_helpers import load_coins, simplify

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch.utils.data import DataLoader, Dataset


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

        # TODO: Change the below to look for (and create) pickled DataFrames
        coins, ts = load_coins('data/pairs/', subset)
        ts = ts.resample(interval, label='right', closed='right', axis=0).asfreq()
        self.raw = ts.dropna(0, 'any')

        self.lag = lag
        self.n_returns = self.raw.shape[0]
        self.train_test_thresh = round(self.n_returns * .8) # hard-coded 80/20 train/test
        self.n_oos = self.n_returns - self.train_test_thresh

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
        ds = self._get_subset(self.train_test_thresh, self.n_samples)
        return DataLoader(ds, batch_size=64, shuffle=True)

    def _get_subset(self, start, end):
        """Helper method for indexing into stored dataset
        :start: Beginning of subset
        :end: End of subset
        :returns: Pandas DataFrame of coin returns
        """
        return self.CryptoReturnsDataset(self.raw.iloc[start : end], self.lag)
