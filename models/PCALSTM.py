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

class PCALSTM(CryptoModel):
    """
    Long/Short-Term Memory Reccurent Netowrk whose training and prediction
        inputs are first transformed according to their Principal Components
    """

    def __init__(self, num_components, lstm_hidden_size):
        """Create the LSTM with PCA-compressed inputs.

        :num_components: Number of components to take when transforming
        :lstm_hidden_size: Size of the hidden dimension in the LSTM predictor

        """

        # Arguments
        self.n_components = num_components
        self.lstm_hid = lstm_hidden_size

        # Torch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        # Hard-coded variables
        self.lstm_epochs = 6
        self.lstm_learning_rate = 0.001

        # Encoder
        self.training_data = np.zeros((3,3,3)) # initialize
        self.pca = PCA(n_components = self.n_components)
        self.lstm = SimpleLSTM(self.n_components, self.lstm_hid, self.training_data.shape[-1])
        self.lstm_training_loss = np.zeros(self.lstm_epochs)

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: Vector of predictions for each of the n_coins.
        """

        for_pca = sample.reshape(sample.shape[0], sample.shape[-1])
        transformed = self.pca.transform(for_pca).reshape(sample.shape[0], 1, -1)
        transformed = torch.tensor(transformed).clone().float().to(self.device)
        print(f'LSTM prediction pca-transformed sample shape: {transformed.shape}')

        # Set the model to evaluation mode
        self.lstm.eval()
        with torch.no_grad():
            return self.lstm(transformed)

    def train(self, training_set):
        """Train, or re-train, the LSTM and AE

        :training_set: DataFrame of training samples

        """
        # Call helper methods
        self._fit_pca(training_set)
        self._train_lstm(training_set)

    def _fit_pca(self, training_set):
        """Helper method to handle fitting the PCA instance

        :training_set: TODO
        :returns: TODO

        """
        self.training_data = training_set.dataset.raw
        self.pca = self.pca.fit(self.training_data)

    def _train_lstm(self, training_set):
        """Helper method to contain the training cycle for the LSTM
        :training_set: PyTorch dataloader for training
        """

        # Set the LSTM to training mode
        self.lstm = SimpleLSTM(self.n_components, self.lstm_hid, self.training_data.shape[-1])
        self.lstm.train()

        ######## Configure the optimiser ########
        optimizer = torch.optim.Adam(
            self.lstm.parameters(),
            lr=self.lstm_learning_rate,
        )
        # Iterate over every batch of sequences
        for epoch in range(self.lstm_epochs):
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

                # Perform the training steps
                output = self.lstm(transformed)        # Step ①
                loss = self.criterion(output, target)  # Step ②
                optimizer.zero_grad()                  # Step ③
                loss.backward()                        # Step ④
                optimizer.step()                       # Step ⑤

            self.lstm_training_loss[epoch] = loss.item()

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        return f"PCA({self.n_components})-LSTM({self.lstm_hid})"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        return f"{self.ae_in}Coins-PCA({self.n_components})_LSTM({self.lstm_hid})"

