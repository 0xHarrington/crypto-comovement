#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
from torch import nn

class CryptoModel(object):
    """Custom interface for ML models easing portfolio simulations"""

    def __init__(self, *args):
        """Create the predictor(s)

        :*args: *shrug*

        """
        raise NotImplemented

    def predict(self, sample):
        """Predict the next out of sample timestep

        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: Vector of predictions for each of the n_coins.

        """
        raise NotImplemented

    def train(self, training_set):
        """Train, or re-train, the predictor(s)

        :training_set: DataFrame of training samples
        :returns: TODO

        """
        raise NotImplemented
