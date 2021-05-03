# Standard imports
import pandas as pd
import numpy as np

def cumret(ts):
    """Calculate cumulative returns for input time series
    :ts: Time series
    :returns: Cumulative returns
    """
    return ((ts + 1).cumprod() - 1) * 100
