#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch import nn

class Results():
    """Wrapper to interface with the predictions from a simulation trial"""

    def __init__(self, oos_dataset, model):
        """Create the custom results class
        :oos_dataset: ground truth for out of sample returns
        """

        # Initialize some values for other methods
        self.cumret = []

        # Save the out of sample returns
        self.y = oos_dataset
        # Save the type of model generating these results
        self.model = model

        # Create the template for the model's returns
        self.y_pred = np.zeros(oos_dataset.shape)

        # index into y_pred for each prediction as it arrives
        self.index = 0

        print(f'Created Results for {self.model.get_fullname()} and OoS dataset of shape {self.y.shape}')

    def portfolio_returns(self):
        """Calculate the cumulative returns over the prediction timeframe
        :returns: Cumulative returns for this portfolio simulation
        """

        # Don't double your work
        if (self.cumret) > 0: return self.cumret

        # convert the predictions to "buy" or "short"
        signs = self.y_pred.copy()
        signs[signs > 0] = 1
        signs[signs < 0] = -1

        # get by-timestep returns as if you invested equally in each coin
        rets = np.multiply(self.y.iloc[1:], signs).sum(axis=1) / self.y.shape[1]
        cumret = (((lstm_ae_returns + 1).cumprod() - 1) * 100)
        self.cumret = cumret

        return cumret

    def add_prediction(self, y_pred):
        """Add prediction and true value to stored results
        :y_pred: Predicted returns at the next timestamp
        """

        # fill in the number of predictions
        n_samples = y_pred.shape[0]
        end = self.index + n_samples
        self.y_pred[self.index:end, :] = y_pred.reshape(n_samples, -1)

        # update the pointer
        self.index += n_samples
        if self.index == self.y.shape[0] - 1:
            print(f'{self.model.get_fullname} Results now full of {self.index} predictions!')

    def get_predictions(self):
        """Reuturn the stored predictions
        :returns: DataFrame of predicted returns
        """
        return self.y_pred[:self.index, :]
