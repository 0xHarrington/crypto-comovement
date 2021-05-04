#!/usr/bin/env python

# Standard imports
import time
import pandas as pd
import numpy as np

# Local Imports
from models.AutoRegressive import AutoRegressive
from models.AutoRegMovingAverage import AutoRegressiveMovingAverage
from models.PCALSTM import PCALSTM
from models.MultivarPCALSTM import MultivarPCALSTM
from models.MultivarAutoEncoderLSTM import MultivarAutoEncoderLSTM
from models.AutoEncoderLSTM import AutoEncoderLSTM
from data.simulation_data import SimulationDataset
from utils.subsets import *
from utils.Results import Results
from utils.plotting import *

"""run_simulation.py: Run a portfolio simulation"""

def simulation(models: dict, ts_data: SimulationDataset, lag: int, retrain_frequency: int):
    """Run one portfolio simulation over the input DataFrame.
    :models: Dictionary of ("Model Name", "Model Object") key-value pairs. Will look for "Model Name".pkl files to load pre-trained models.
    :ts_data: DataFrame of the time series over which you want to train and test.
    :returns: Dictionary of ("Model Name", "Model Predictions") across the testing time period
    """

    # Grab the dimensions and initialize the prediction husk
    oos_ds = ts_data.get_out_of_sample().dataset.raw
    oos_size, n_coins = oos_ds.shape
    buy_and_hold = np.zeros((oos_size, 1))
    results = {}
    for name, model in models.items():
        results[name] = Results(oos_ds, model, lag)

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
                tic = time.perf_counter()
                model.train(ds)
                toc = time.perf_counter()
                print(f"\tTrained {name} in {toc - tic:0.4f} seconds")

            # Perform and store the prediction!
            #     NOTE: Assuming predicting 1-at-a-time given hard-coded reshape
            prediction = model.predict(oos_data)
            results[name].add_prediction(prediction.reshape(1, -1))

    print('======= Predicted everything! =======')

    cret = oos_ds.mean(axis=1)

    return results, oos_ds.mean(axis=1).iloc[lag:]

if __name__ == "__main__":
    subset = one_yr
    interval = '1D'
    lag = 1 # not ready for not 1
    latent_dim = 2
    retrain_frequency = 2
    dataset = SimulationDataset(subset, interval, lag)

    # Menu of models
    menu = {
        "AR": "AutoRegressive(latent_dim)",
        'ARMA': "AutoRegressiveMovingAverage(latent_dim)",
        #  'AELSTM': "AutoEncoderLSTM(len(subset), latent_dim, 4)",
        #  'PCALSTM': "PCALSTM(latent_dim, 4)",
        'MvarAELSTM': "MultivarAutoEncoderLSTM(len(subset), latent_dim, 4)",
        'MvarPCALSTM': "MultivarPCALSTM(latent_dim, 4)",
    }
    # Model order for the kitchen
    model_order = [ 0, 0, 0,20]

    # Initialize and populate models dict
    models = {}
    for i, (name, model_str) in enumerate(menu.items()):
        print(i, name, model_str)
        n = model_order[i]
        if n == 0: continue
        if n == 1: models[name] = eval(model_str)
        else:
            for i in range(n):
                models[name + str(i + 1)] = eval(model_str)

    reults, buy_and_hold = simulation(models, dataset, lag, retrain_frequency)

    print("========== SIMULATION RETURNS ==========")

    for name, p in reults.items():
        mname = p.get_model_name()
        ret = p.portfolio_returns()
        print(f'{mname}:\t{round(ret[-1], 4)}%')

    # Plot the simulation results
    plot_portfolio_sims(reults, subset, buy_and_hold)
