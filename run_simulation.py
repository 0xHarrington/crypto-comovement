#!/usr/bin/env python

# Standard imports
import pickle
import pandas as pd
import numpy as np
# Tiemstamping
import time
from datetime import datetime

# Local Imports
from models.LASSO import LASSO
from models.AutoRegressive import AutoRegressive
from models.AutoRegMovingAverage import AutoRegressiveMovingAverage
from models.MultivarPCALSTM import MultivarPCALSTM
from models.MultivarAutoEncoderLSTM import MultivarAutoEncoderLSTM
from models.MultivarMultiModelAutoEncoderLSTM import MultivarMultiModelAutoEncoderLSTM
from models.MultivarAutoEncoderFFNN import MultivarAutoEncoderFFNN
from data.simulation_data import SimulationDataset
from utils.subsets import *
from utils.Results import Results
from utils.simulations import cumret
from utils.plotting import *

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
    lag = ts_data.lag
    buy_and_hold = np.zeros((oos_size, 1))
    results = {}
    for name, model in models.items():
        results[name] = Results(oos_ds, model, lag)

    print('============== Beginning predictions! ==============')
    for oos_sample, (oos_data, target) in enumerate(ts_data.get_out_of_sample()):

        # Grab new data if retrain is necessary
        retrain = (oos_sample % retrain_frequency == 0)
        if retrain:
            ds = ts_data.get_training(oos_sample)
            print(f'~~~~~ Retraining models for {oos_sample}\'th prediction ~~~~~')

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

    print('=============== Predicted everything! ===============')

    cret = oos_ds.mean(axis=1)

    return results, oos_ds.mean(axis=1).iloc[lag:]

if __name__ == "__main__":
    subset = two_yr
    interval = '1D'
    lag = 1 # not ready for not 1
    latent_dim = 2
    retrain_frequency = 1
    train_test_threshold = 0.8
    dataset = SimulationDataset(subset, interval, lag, train_test_threshold)
    PICKLE_RESULTS = True

    n_multi_models = 20

    # Menu of standard models
    menu = {
        "AR": "AutoRegressive(lag)",
        'ARMA': "AutoRegressiveMovingAverage(lag)",
        'LASSO': "LASSO()",
        f'MvarMM{(n_multi_models)}AELSTMs': "MultivarMultiModelAutoEncoderLSTM(len(subset), latent_dim, 4, n_multi_models)",
        'MvarAELSTM': "MultivarAutoEncoderLSTM(len(subset), latent_dim, 4)",
        'MvarPCALSTM': "MultivarPCALSTM(latent_dim, 4)",
        'MvarAEFFNN': "MultivarAutoEncoderFFNN(len(subset), latent_dim, 15)"
    }
    # Model order for the kitchen
    model_order = [1,1,1,0,3,3,3]

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

    ####################################
    ############ RUN THE SIM ###########
    ####################################

    results, buy_and_hold = simulation(models, dataset, retrain_frequency)

    print("========== SIMULATION RETURNS ==========")

    for name, p in results.items():
        mname = p.get_model_name()
        ret = p.portfolio_returns()
        print(f'{mname}:\t{round(ret[-1], 4)}%')

        if PICKLE_RESULTS:
            now = datetime.now()
            tformat = "%Y-%m-%d_%H:%M:%S"
            fname = f'{now.strftime(tformat)}-retrain({retrain_frequency})-{p.get_model_fname()}.pkl'
            f = open(f'results/pickle/{fname}', 'wb')
            pickle.dump(p, f)
            print(f'Results dumped to {fname}!')
            time.sleep(1)

    i = list(results.values())[0].portfolio_returns().index
    bnh = cumret(buy_and_hold[-len(i):])
    print(f'----- Buy and Hold:\t{round(bnh[-1], 2)}% -----')

    # Plot the simulation results
    plot_portfolio_sims(results, subset, bnh)
