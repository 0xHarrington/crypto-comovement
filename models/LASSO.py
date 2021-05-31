#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch import nn

# Using sklearn's LASSO implementation
from sklearn.linear_model import Lasso

# Local Files
from models.model_interface import CryptoModel

class LASSO(CryptoModel):
    """Wrapper around the sklearn LASSO class"""

    def __init__(self, alpha=0.1, warm_start=True, verbose_training=False):
        """Create the LASSO model.

        :input_size: Input size to the AutoEncoder, should be n_coins

        """

        # Arguments
        self.alpha = alpha
        self.verbose_training = verbose_training

        self.model = Lasso(alpha=alpha, fit_intercept=True, warm_start=warm_start)

        # set the default plotting color
        self.set_plotting_color()

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: [batch_size, 1, n_coins] Tensor of predictions
        """
        n_samp, _, n_features = sample.shape
        yhat = self.model.predict(sample.reshape((n_samp, n_features)))
        if self.verbose_training:
            print(f'prediction: {yhat}')
        return yhat

    def train(self, training_set):
        """Train, or re-train, the LSTM and AE

        :training_set: DataFrame of training samples

        """
        X, Y = [], []
        for data, target in training_set:
            X.append(data.numpy())
            Y.append(target.numpy())

        X = np.vstack(X)
        Y = np.vstack(Y)
        n_samples, _, n_features = X.shape
        X = X.reshape((n_samples, n_features))
        Y = Y.reshape((n_samples, n_features))
        self.model.fit(X, Y)

        #TODO: print out that coefficients (or at least num of) that the L1 normalization leaves
        coef = self.model.coef_
        all_zeros = np.isin([0,-0], coef)
        if self.verbose_training:
            print(f'All zeros? {all_zeros}')
            print(f'Coefs? {coef.shape}')
            if np.isin(False, all_zeros):
                print(self.model.coef_)
                print(type(self.model.coef_))
                print(self.model.coef_.shape)

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        return f"LASSO_alpha-{self.alpha}"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        return f"LASSO_alpha-{self.alpha}"

    def needs_retraining(self):
        """Does this model need regular retraining while forecasting?
        :returns: bool
        """
        return True

    def set_plotting_color(self, color="#FCB97D"):
        """Set color used for plotting
        :color: Hex value string
        """
        self.color = color

    def get_plotting_color(self):
        """return color for graphing distinction
        :returns: str of color
        """
        return self.color
