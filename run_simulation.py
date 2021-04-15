#!/usr/bin/env python
import pandas as pd
import numpy as np

"""run_simulation.py: Run a portfolio simulation"""


def simulation(models: dict, ts_data: pd.core.frame.DataFrame):
    """Run one portfolio simulation over the input DataFrame.
    :models: Dictionary of ("Model Name", "Model Object") key-value pairs. Will look for "Model Name".pkl files to load pre-trained models.
    :ts_data: DataFrame of the time series over which you want to train and test.
    :returns: Dictionary of ("Model Name", "Model Predictions") across the testing time period
    """

    print("We in here!")
    return

if __name__ == "__main__":

    models = {}


    #  do I want to leverage pre-trained models?
    #  if so, hey pickled models directory, are any of you the type we want?
    #  are any of you trained on this dataset?
    #  sweet, let me load all the ones we need

    #  any leftover we haven't trained yet?
    #  okay, let's get you trained and ready

    #  now, let's start making some predictions!
    #  prep where we store the predictions...
    #  now, start generating predictions!
    #  if we've gone some specific number of predictions without a retrain, retrain!

    #  now, store the results in which we're interested.

    x = np.arange(3)

    simulation(models, pd.DataFrame(x))
