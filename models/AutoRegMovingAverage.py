#!/usr/bin/env python

# Standard imports
import warnings
import pandas as pd
import numpy as np

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Local Files
from models.model_interface import CryptoModel

class AutoRegressiveMovingAverage(CryptoModel):
    """Univariate ARMA Model: latent variable model using only the past time steps and their moving average"""

    def __init__(self, lag: int):
        """Create the AR model

        :lag: Number of timesteps to look back

        """
        self.lag = lag
        self.models = {}

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: [batch_size, 1, n_coins] Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: [batch_size, 1, n_coins] Tensor of predictions
        """

        n_samples, _, features = sample.shape
        n_coins = len(self.models.keys())
        ret = np.zeros((sample.shape[0], 1, n_coins))

        for i in range(n_samples):
            new_models = {}
            for j, (coin, model) in enumerate(self.models.items()):
                appended = model.append(np.array([ sample[i, 0, j] ]))
                ret[i, 0, j] = appended.forecast()
                new_models[coin] = appended
            self.models = new_models
        return ret

    def train(self, training_set):
        """Train, or re-train, the AR models
        :training_set: DataFrame of training samples
        :returns: TODO
        """

        self.training_data = training_set.dataset.raw
        self.models = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for coin, ts in self.training_data.iteritems():
                self.models[coin] = SARIMAX(ts, order = (self.lag, 0, self.lag), trend='c').fit(disp=False)

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string

        """
        return f"Auto-Regressive Moving-Average ({self.lag}, {self.lag}) latent variable model"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores

        """
        return f"ARMA({self.lag}-{self.lag})"

    def needs_retraining(self):
        """Does this model need regular retraining while forecasting?
        :returns: bool
        """
        return False
