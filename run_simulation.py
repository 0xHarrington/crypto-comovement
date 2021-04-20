#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Local Imports
from models.AutoRegressive import AutoRegressive
from models.PCALSTM import PCALSTM
from models.AutoEncoderLSTM import AutoEncoderLSTM
from data.simulation_data import SimulationDataset
from utils.subsets import *

"""run_simulation.py: Run a portfolio simulation"""

def simulation(models: dict, ts_data: SimulationDataset):
    """Run one portfolio simulation over the input DataFrame.
    :models: Dictionary of ("Model Name", "Model Object") key-value pairs. Will look for "Model Name".pkl files to load pre-trained models.
    :ts_data: DataFrame of the time series over which you want to train and test.
    :returns: Dictionary of ("Model Name", "Model Predictions") across the testing time period
    """

    starting_index = 0

    for name, model in models.items():
        ds = ts_data.get_training(starting_index)

        model.train(ds)
        print(f"Trained {name}!")

        for data, target in ts_data.get_out_of_sample():
            prediction = model.predict(data).shape
            break
    print('Predicted everything!')

    return

if __name__ == "__main__":
    subset = one_yr
    interval = '1D'
    lag = 1
    latent_dim = 2
    dataset = SimulationDataset(subset, interval, 1)

    models = {}
    models['AELSTM'] = AutoEncoderLSTM(len(subset), latent_dim, 4)
    models['PCALSTM'] = PCALSTM(latent_dim, 4)
    models['AR'] = AutoRegressive(latent_dim)

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

    simulation(models, dataset)
