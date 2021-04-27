# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_portfolio_sims(results: dict, subset = [], style='ggplot'):
    """Plot a collection of cumulative returns from portfolio simulations

    :results: Dict of (Name, Cumulative_Returns) key/value pairs
    :subset: List of coin tickers from which the Results trade
    :style: optional plotting style (default: 'ggplot')

    """

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12,8))
    for name, res in results.items():
        plt.plot(res.portfolio_returns()*100, label=name, color=res.get_model_color())

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(f"{len(subset)}-Coin Portfolio Simulation - Latent Factor Models")
    plt.legend()
    plt.show()
