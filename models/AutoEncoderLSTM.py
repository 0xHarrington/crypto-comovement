#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch import nn

# Local Files
from models.model_interface import CryptoModel
from models.SimpleLSTM import SimpleLSTM

class AutoEncoderLSTM(CryptoModel):
    """Custom interface for ML models easing portfolio simulations"""

    class AutoEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # ================================================================

    def __init__(self, ae_input_size, ae_hidden_size, lstm_hidden_size):
        """Create the LSTM with AutoEncoder-compressed inputs.

        :ae_input_size: Input size to the AutoEncoder, should be n_coins
        :ae_hidden_size: Size of the middle layer in the AutoEncoder
        :lstm_hidden_size: Size of the hidden dimension in the LSTM predictor

        """

        # Arguments
        self.ae_in = ae_input_size
        self.ae_hid = ae_hidden_size
        self.lstm_hid = lstm_hidden_size

        # Torch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        # Hard-coded variables
        self.ae_epochs = 200
        self.ae_learning_rate = 0.001
        self.lstm_epochs = 6
        self.lstm_learning_rate = 0.001

        # Encoder
        self.ae = self.AutoEncoder(ae_input_size, ae_hidden_size)
        self.ae_training_loss = np.zeros(self.ae_epochs)
        self.lstm = SimpleLSTM(self.ae_hid, self.lstm_hid, ae_input_size)
        self.lstm_training_loss = np.zeros(self.lstm_epochs)

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: Vector of predictions for each of the n_coins.
        """

        # Set the model to evaluation mode
        self.lstm.eval()
        with torch.no_grad():
            transformed = self.ae.encoder(sample)
            return self.lstm(transformed)

    def train(self, training_set):
        """Train, or re-train, the LSTM and AE

        :training_set: DataFrame of training samples

        """
        self._train_ae(training_set)
        self._train_lstm(training_set)

    def _train_ae(self, training_set):
        """Helper method to contain the training cycle for the AutoEncoder
        :training_set: PyTorch dalaloader training
        """

        # Overwrite cached AE % losses
        self.ae = self.AutoEncoder(self.ae_in, self.ae_hid)
        self.training_loss = np.zeros(self.ae_epochs)

        ######## Configure the optimiser ########
        optimizer = torch.optim.Adam(
            self.ae.parameters(),
            lr=self.ae_learning_rate,
        )

        ######## Run the training loops ########
        for epoch in range(self.ae_epochs):
            for data in training_set:
                x, _ = data
                x = x.clone().float().to(self.device)
                x = x.view(x.size(0), -1)

                # =================== forward =====================
                output = self.ae(x)  # feed <x> (for std AE) or <x_bad> (for denoising AE)
                loss = self.criterion(output, x.data)

                # =================== backward ====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # =================== record ========================
            self.ae_training_loss[epoch] = loss.item()

    def _train_lstm(self, training_set):
        """Helper method to contain the training cycle for the LSTM
        :training_set: PyTorch dataloader for training
        """

        # Set the LSTM to training mode
        self.lstm = SimpleLSTM(self.ae_hid, self.lstm_hid, self.ae_in)
        self.lstm.train()

        ######## Configure the optimiser ########
        optimizer = torch.optim.Adam(
            self.lstm.parameters(),
            lr=self.ae_learning_rate,
        )
        # Iterate over every batch of sequences
        for epoch in range(self.lstm_epochs):
            for data, target in training_set:

                # reshape target per pytorch warnings
                t_shape = target.shape
                target = target.reshape(t_shape[0], 1, t_shape[1])

                # Convert
                data = self.ae.encoder(data.clone().float().to(self.device))
                target = target.clone().float().to(self.device)

                # Perform the training steps
                output = self.lstm(data)               # Step ①
                loss = self.criterion(output, target)  # Step ②
                #  print(f'LSTM training: output shape {output.shape}, target shape {target.shape}')
                optimizer.zero_grad()                  # Step ③
                loss.backward()                        # Step ④
                optimizer.step()                       # Step ⑤

            self.lstm_training_loss[epoch] = loss.item()

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        return f"AE({self.ae_hid})-LSTM({self.lstm_hid})"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        return f"{self.ae_in}Coins-AE({self.ae_hid})_LSTM({self.lstm_hid})"

