#!/usr/bin/env python

# Standard imports
import time
import pandas as pd
import numpy as np

# Local Imports
from run_simulation import simulation
from models.LASSO import LASSO
from data.simulation_data import SimulationDataset
from utils.subsets import *
from utils.Results import Results
from utils.plotting import *

"""ffnn_tuning_sim.py: running simulations to determine ideal hidden dimension for Feed-Forward Neural Networks"""

if __name__ == "__main__":
    #  subset = one_yr
    subset = two_yr
    interval = '1D'
    lag = 1 # not ready for not 1
    latent_dim = 2
    retrain_frequency = 5
    dataset = SimulationDataset(subset, interval, lag)

    # Initialize and populate models dict
    models = {}

    ####################################
    ########## TESTING MODELS ##########
    ####################################

    #  lasso_alpha = np.arange(1, 10, 5) / 10
    #  n_each = 2

    lasso_alpha = np.arange(1, 10) / 10
    n_each = 20

    # Testing best Feed Forward Neural Net hidden size
    for i in lasso_alpha:
        for j in range(n_each):
            models[f"l-{i}-{j}"] = LASSO(i)
            #  models[f"l-{i}-{j}"] = LASSO(i, verbose_training=True)

    ####################################
    ############ RUN THE SIM ###########
    ####################################

    results, buy_and_hold = simulation(models, dataset, retrain_frequency)

    print("========== SIMULATION RETURNS ==========")

    for name, p in results.items():
        mname = p.get_model_name()
        ret = p.portfolio_returns()
        print(f'{mname}:\t{round(ret[-1], 4)}%')

    # Plot the simulation results
    plot_return_distributions(results, subset)
