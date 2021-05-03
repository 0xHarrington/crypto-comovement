# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# local files
from utils.simulations import cumret

def plot_portfolio_sims(results: dict, subset=[], buy_and_hold=np.zeros((1,1)), style='ggplot'):
    """Plot a collection of cumulative returns from portfolio simulations

    :results: Dict of (Name, Cumulative_Returns) key/value pairs
    :subset: List of coin tickers from which the Results trade
    :buy_and_hold: Results from simply buying equal amounts of each asset and holding
    :style: optional plotting style (default: 'ggplot')

    """

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12,8))
    for name, res in results.items():
        plt.plot(res.portfolio_returns(), label=name, color=res.get_model_color())

    if buy_and_hold.shape != (1,1):
        i = list(results.values())[0].portfolio_returns().index
        bnh = cumret(buy_and_hold[-len(i):])
        plt.plot(i, bnh, '--', label='B&H', color='black')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(f"{len(subset)}-Coin Portfolio Simulation - Latent Factor Models")
    if len(results.keys()) < 10:
        plt.legend()
    plt.show()
