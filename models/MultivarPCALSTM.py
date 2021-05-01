#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch import nn

# SKlearn PCA
from sklearn.decomposition import PCA

# Local Files
from models.SimpleLSTM import SimpleLSTM
from models.model_interface import CryptoModel

class MultivarPCALSTM(CryptoModel):
    """
    Long/Short-Term Memory Reccurent Netowrk whose training and prediction
        inputs are first transformed according to their Principal Components
    """

    def __init__(self, num_components, lstm_hidden_size, verbose_training = False):
        """Create the LSTM with PCA-compressed inputs.

        :num_components: Number of components to take when transforming
        :lstm_hidden_size: Size of the hidden dimension in the LSTM predictor

        """

        # Arguments
        self.n_components = num_components
        self.lstm_hid = lstm_hidden_size
        self.verbose_training = verbose_training

        # Torch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        # Hard-coded variables
        self.lstm_epochs = 6
        self.lstm_learning_rate = 0.001

        # Transformer
        self.training_data = np.zeros((3,3,3)) # initialize
        self.pca = PCA(n_components = self.n_components)

        # Initialize multivariate LSTMs
        self.lstm_arr = [0] * self.training_data.shape[-1]
        for i in range(len(self.lstm_arr)):
            self.lstm_arr[i] = SimpleLSTM(self.n_components, self.lstm_hid, 1)
        self.lstm_training_loss = np.zeros(self.lstm_epochs)

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: [batch_size, 1, n_coins] Tensor of predictions
        """

        for_pca = sample.reshape(sample.shape[0], sample.shape[-1])
        transformed = self.pca.transform(for_pca).reshape(sample.shape[0], 1, -1)
        transformed = torch.tensor(transformed).clone().float().to(self.device)

        ret_list = []
        for i in range(len(self.lstm_arr)):
            # Set the model to evaluation mode
            self.lstm_arr[i].eval()
            with torch.no_grad():
                ret = self.lstm_arr[i](transformed)
                ret_list.append(ret)

        return torch.cat(ret_list, 2)

    def train(self, training_set):
        """Train, or re-train, the LSTM and AE

        :training_set: DataFrame of training samples

        """

        # Re-initialize multivariate LSTMs
        self.training_data = training_set.dataset.raw
        self.lstm_arr = [0] * self.training_data.shape[-1]
        for i in range(len(self.lstm_arr)):
            self.lstm_arr[i] = SimpleLSTM(self.n_components, self.lstm_hid, 1)
        self.lstm_training_loss = np.zeros(self.lstm_epochs)

        # Call helper methods
        self._fit_pca(training_set)
        self._train_lstm(training_set)

    def _fit_pca(self, training_set):
        """Helper method to handle fitting the PCA instance
        :training_set: input CryptoModel from which to extract raw time series df
        """

        self.pca = self.pca.fit(self.training_data)
        if self.verbose_training:
            print(f'{self.get_fullname()} finished fitting PCA')

    def _train_lstm(self, training_set):
        """Helper method to contain the training cycle for the LSTM
        :training_set: PyTorch dataloader for training
        """

        ######## Configure the optimisers ########
        optimisers = [0] * self.training_data.shape[-1]
        for i in range(len(self.lstm_arr)):
            optimisers[i] = torch.optim.Adam(
                self.lstm_arr[i].parameters(),
                lr=self.lstm_learning_rate,
            )

            # Set the LSTMs to training mode
            self.lstm_arr[i].train()

        # Iterate over every batch of sequences
        for epoch in range(self.lstm_epochs):

            # Verbose Debugging
            if self.verbose_training and (epoch % (self.lstm_epochs // 3) == 0):
                print(f'{self.get_fullname()} at LSTM training epoch {epoch}')

            for data, target in training_set:

                # reshape target per pytorch warnings
                t_shape = target.shape
                target = target.reshape(t_shape[0], 1, t_shape[1])

                # Convert
                d_shape = data.shape
                for_pca = data.reshape(d_shape[0], d_shape[-1])
                transformed = self.pca.transform(for_pca).reshape(d_shape[0], 1, -1)
                transformed = torch.tensor(transformed).clone().float().to(self.device)
                target = target.clone().float().to(self.device)

                # Perform the training steps for each of the models
                for i in range(len(self.lstm_arr)):
                    t = target[:,:,i].reshape(-1, 1, 1)
                    output = self.lstm_arr[i](transformed)  # Step ①
                    loss = self.criterion(output, t)        # Step ②
                    optimisers[i].zero_grad()               # Step ③
                    loss.backward()                         # Step ④
                    optimisers[i].step()                    # Step ⑤

            self.lstm_training_loss[epoch] = loss.item()

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        return f"Multivariate PCA({self.n_components})-LSTM({self.lstm_hid},{self.lstm_epochs})"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        return f"MVar_{self.training_data.shape[-1]}Coins_PCA({self.n_components})-LSTM({self.lstm_hid},{self.lstm_epochs})"

    def needs_retraining(self):
        """Does this model need regular retraining while forecasting?
        :returns: bool
        """
        return True

    def get_plotting_color(self):
        """return color for graphing distinction
        :returns: str of color
        """
        return "#76a962"
