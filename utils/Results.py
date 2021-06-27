# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch import nn
from sklearn.metrics import mean_squared_error

# local files
from utils.simulations import cumret

class Results():
    """Wrapper to interface with the predictions from a simulation trial"""

    def __init__(self, oos_dataset, model, lag):
        """Create the custom results class
        :oos_dataset: ground truth for out of sample returns
        """

        # Initialize some values for other methods
        self.cumret = []
        self.returns = []

        # Save arguments
        self.y = oos_dataset
        self.model = model
        self.lag = lag

        # Create the template for the model's returns
        self.y_pred = np.zeros((oos_dataset.shape[0]-1, oos_dataset.shape[1]))

        # index into y_pred for each prediction as it arrives
        self.index = 0

        print(f'Created Results for {self.model.get_fullname()} and OoS dataset of shape {self.y.shape}')

    def get_buy_and_hold(self):
        """Returns the individual returns from investing evenly across all coins in the simulation
        :returns: TODO

        """
        return self.y.iloc[self.lag:].mean(axis=1)

    def get_portfolio_returns(self):
        """Calculate the individual returns over the prediction timeframe
        :returns: By-Period returns for this portfolio simulation
        """

        # Don't double your work
        #  if (len(self.returns)) > 0: return self.returns

        # convert the predictions to "buy" or "short"
        signs = self.y_pred.copy()
        signs[signs > 0] = 1
        signs[signs < 0] = -1

        # get by-timestep returns as if you invested equally in each coin
        true = self.y.iloc[self.lag:]
        rets = np.multiply(true, signs).mean(axis=1)
        self.returns = rets

        return rets

    def cumulative_portfolio_returns(self):
        """Returns the cumulative returns of this portfolio simulation
        :returns: np array
        """

        # Don't double the work
        if (len(self.cumret)) > 0: return self.cumret

        rets = self.get_portfolio_returns()
        cret = cumret(rets)
        self.cumret = cret

        return cret

    def get_sharpe(self, periods_per_day=1):
        """Returns the sharpe of the portfolio reutrns
        :periods_per_day: How many samples in this Results object equal one day
        """

        # Calculate the risk free return over the period
        risk_free_rate_apy = 0.03
        daily_rfr = risk_free_rate_apy / 365
        n_days = len(self.y_pred) / periods_per_day
        rfr = daily_rfr * n_days

        # Calculate the portfolio_returns and their standard deviation
        rets = self.get_portfolio_returns()
        crets = self.cumulative_portfolio_returns()
        std = np.std(rets)
        sim_return = (crets.iloc[-1] - 1) / 100

        return (sim_return - rfr) / std


    def get_rmse(self):
        """Returns the average RMSE across all predictions
        """

        true = self.y.iloc[self.lag:]
        return mean_squared_error(true.values, self.y_pred)

    def add_prediction(self, y_pred):
        """Add prediction and true value to stored results
        :y_pred: Predicted returns at the next timestamp
        """

        # fill in the number of predictions
        n_samples = y_pred.shape[0]
        end = self.index + n_samples
        self.y_pred[self.index:end, :] = y_pred.reshape(n_samples, -1)

        # update the pointer
        self.index += n_samples
        if self.index == self.y.shape[0] - 1:
            print(f'{self.model.get_fullname()} Results now full of {self.index} predictions')

    def get_predictions(self):
        """Reuturn the stored predictions
        :returns: DataFrame of predicted returns
        """
        return self.y_pred[:self.index, :]

    def get_model_name(self):
        """Get the full name of the model that generated these results
        :returns: str
        """
        return self.model.get_fullname()

    def get_model_fname(self):
        """Get the full name of the model that generated these results
        :returns: str
        """
        return self.model.get_filename()

    def get_model_color(self):
        """Returns saved model's plotting color
        :returns: hex str
        """
        return self.model.get_plotting_color()
