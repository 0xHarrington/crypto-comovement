#!/usr/bin/env python

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
        """
        raise NotImplemented

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        raise NotImplemented

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        raise NotImplemented
