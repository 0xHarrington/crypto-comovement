import sys
import os
import pandas as pd
import numpy as np

# Subsets
from utils.subsets import *

# Generate log returns of ts
def log_ret(ts):
    np.seterr(divide='ignore')
    logged = np.log(ts).replace(-np.inf, 0)
    diffed = logged.diff()
    return pd.DataFrame(diffed.dropna(how='all'))

# -----------------------------------------------------------------------------

# Resample to given interval
def simplify(ts, interval):
    return ts.resample(interval, label='right', closed='right', axis=0).asfreq()

# -----------------------------------------------------------------------------

def load_coins(import_path = 'pairs/', subset = ['BTC', 'ETH']):
    ''' Return pandas dataframe of all .parquet coin values.
    '''
    coins = _make_coins_dict(import_path, subset)
    log_returns = _make_returns_df(coins, subset)
    return coins, log_returns

def coin_subset_from_n_days(ts: pd.core.frame.DataFrame, n_days: int):
    """Get all the coins from ts
    :ts: DataFrame of coin returns
    :n_days: Minimum number of days during which the coin has been listed.

    :returns: list of coin tickers
    """

    day_ret = simplify(ts, '1D')
    ret = []
    for c in day_ret.columns:
        nulls = day_ret[c].isnull().sum()
        lengs = day_ret[c].shape[0]
        days = lengs - nulls
        if days > n_days:
            ret.append(c)

    return ret

# =============================================================================
# =============================================================================

# Turn dict of {coin: filename} into pd df
def _make_returns_df(coins, subset = ['BTC', 'ETH']):
    # Aggregate all 'closes' to one df
    prices_df = []
    for c, f in coins.items():
        prices_df = _append_col(f, prices_df)

    # Changing the column names to fit the subset
    top_coins = []
    for c in subset:
        if c in prices_df.columns:
            top_coins.append(c)

    prices_df = prices_df.reindex(columns=top_coins)
    return log_ret(prices_df)

def _make_coins_dict(import_path, subset):
    # Gather all fileneames with tether or in top-50
    coins, ref = {}, subset
    for dirname, _, filenames in os.walk(import_path):
        for filename in filenames:
            c1, c2 = _parse_filename(filename)

            # Find only coins in the top X
            if c1 in top_50 and c2 in top_50:

                # screen for blacklist
                if c1 in blacklist or c2 in blacklist: continue

                # Test if stable pair
                if c1 in stable and c2 not in stable:
                    stab, coin = c1, c2
                elif c2 in stable and c1 not in stable:
                    stab, coin = c2, c1
                else: continue

                # Check if already saved, save the more popular stable coin pair
                if coin in coins.keys():
                    oc1, oc2 = _parse_filename(coins[coin])
                    if oc1 in stable and oc2 not in stable:
                        ostab, ocoin = oc1, oc2
                    elif oc2 in stable and oc1 not in stable:
                        ostab, ocoin = oc2, oc1
                    old_stable = stable.index(ostab)
                    curr_stable = stable.index(stab)
                    if curr_stable < old_stable:
                        coins[coin] = filename
                else:
                    coins[coin] = filename

    # Add full path in final pass
    for c, f in coins.items():
        coins[c] = import_path + f

    # TODO: Toggle to see top coins without a stable pair
    if False:
        no_stable = []
        for c in subset: 
            if c in stable: continue
            if c not in coins.keys():
                print("#{}:\t{}".format(top_50.index(c), c))
                no_stable.append(c)

    return coins

# Return df appended with new df's column
def _append_col(filename, df = [], col = 'close'):
    f = filename.split('/')[-1].split('.')[0]
    c1, c2 = f.split('-')

    # Grab names of coins in pair
    if c1 in stable and c2 not in stable:
        stab, coin = c1, c2
    elif c2 in stable and c1 not in stable:
        stab, coin = c2, c1

    # print("Adding {}/{}...".format(coin, stab))
    ts = pd.read_parquet(filename)[col]
    ts.name = coin
    ts.columns = coin

    if isinstance(df, list): 
        df = ts
        df.columns = [coin]
    elif isinstance(df, pd.Series):
        ts = pd.Series(ts, name=coin)
        df = pd.concat([df, ts], axis=1)
    else:
        df = df.join(ts, how='outer')

    return df

def _parse_filename(f):
    names = f.split('.')[0].split('-')
    return names[0], names[1]
