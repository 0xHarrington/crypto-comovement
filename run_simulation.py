#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Local Imports
from models.AutoRegressive import AutoRegressive
from models.AutoRegMovingAverage import AutoRegressiveMovingAverage
from models.PCALSTM import PCALSTM
from models.AutoEncoderLSTM import AutoEncoderLSTM
from data.simulation_data import SimulationDataset
from utils.subsets import *
from utils.Results import Results

"""run_simulation.py: Run a portfolio simulation"""

def simulation(models: dict, ts_data: SimulationDataset, retrain_frequency: int):
    """Run one portfolio simulation over the input DataFrame.
    :models: Dictionary of ("Model Name", "Model Object") key-value pairs. Will look for "Model Name".pkl files to load pre-trained models.
    :ts_data: DataFrame of the time series over which you want to train and test.
    :returns: Dictionary of ("Model Name", "Model Predictions") across the testing time period
    """

    # Grab the dimensions and initialize the prediction husk
    oos_ds = ts_data.get_out_of_sample().dataset.raw
    oos_size, n_coins = oos_ds.shape
    predictions = {}
    results = {}
    for name, model in models.items():
        predictions[name] = np.ones((oos_size, n_coins)) * 9 # easier to debug
        results[name] = Results(oos_ds, model)

    print('======= Beginning predictions! =======')
    for oos_sample, (oos_data, target) in enumerate(ts_data.get_out_of_sample()):

        # Grab new data if retrain is necessary
        retrain = (oos_sample % retrain_frequency == 0)
        if retrain:
            ds = ts_data.get_training(oos_sample)
            print(f'~~ Retraining models for {oos_sample}\'th prediction ~~')

        for name, model in models.items():
            #  Re-train if necessary
            if (retrain and model.needs_retraining()) or (oos_sample == 0):
                model.train(ds)
                print(f"\tTrained {name}!")

            # Perform and store the prediction!
            #     NOTE: Assuming predicting 1-at-a-time given hard-coded reshape
            prediction = model.predict(oos_data)
            predictions[name][oos_sample, :] = prediction.reshape(1, -1)
            results[name].add_prediction(prediction.reshape(1, -1))

    print('======= Predicted everything! =======')

    return results

if __name__ == "__main__":
    subset = one_yr
    interval = '1D'
    lag = 1
    latent_dim = 2
    retrain_frequency = 10
    dataset = SimulationDataset(subset, interval, 1)

    models = {}
    models['AELSTM'] = AutoEncoderLSTM(len(subset), latent_dim, 4)
    models['PCALSTM'] = PCALSTM(latent_dim, 4)
    models['AR'] = AutoRegressive(latent_dim)
    models['ARMA'] = AutoRegressiveMovingAverage(latent_dim)

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

    predictions = simulation(models, dataset, retrain_frequency)

    for name, p in predictions.items():
        preds = p.get_predictions()
        print(f'{name} made predictions of shape {preds.shape}')
        print(f'\tTheir tail: {preds[2:, :]}')
        print()

