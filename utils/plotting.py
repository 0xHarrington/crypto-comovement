# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import kde

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
    plotted_model_types = []
    for name, res in results.items():
        if name not in plotted_model_types:
            plotted_model_types.append(name)
            plt.plot(res.portfolio_returns(), label=name, color=res.get_model_color())
        else:
            plt.plot(res.portfolio_returns(), color=res.get_model_color())

    if buy_and_hold.shape != (1,1):
        i = list(results.values())[0].portfolio_returns().index
        plt.plot(i, buy_and_hold, '--', label='B&H', color='black')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(f"{len(subset)}-Coin Portfolio Simulation")
    print(f'{len(plotted_model_types)} unique models?')
    if len(plotted_model_types) < 10:
        plt.legend()
    plt.show()

def plot_return_distributions(results: dict, subset=[], style='ggplot'):
    """Plot the KDE plots of each of the dimension's final returns

    :results: Dict of (Name, Cumulative_Returns) key/value pairs
    :subset: List of coin tickers from which the Results trade
    :style: optional plotting style (default: 'ggplot')

    """

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12,8))

    # Create new dict of all the total returns from each dimension size
    rets = {}
    for name in results.keys():
        print(name)
        _, dim, num = name.split('-')
        rets[dim] = []
    min_x, max_x = np.Inf, np.NINF
    for name, res in results.items():
        _, dim, num = name.split('-')
        ret = res.portfolio_returns()[-1]
        rets[dim].append(ret)

        # update the linespace boundaries
        if ret > max_x: max_x = int(ret)
        if ret < min_x: min_x = int(ret)

    # plot the kde
    for key, arr in rets.items():
        x = np.linspace(min_x-1, max_x+1, 50 * (max_x - min_x))
        density = kde.gaussian_kde(arr)
        y = density(x)
        plt.plot(x,y, label=f"Dim-{key}")

    plt.title(f"{len(subset)}-Coin Portfolio Simulation - Dimension Comparison")
    plt.legend()
    plt.show()
