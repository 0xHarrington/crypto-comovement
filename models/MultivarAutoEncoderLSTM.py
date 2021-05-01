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

class MultivarAutoEncoderLSTM(CryptoModel):
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

    def __init__(self, ae_input_size, ae_hidden_size, lstm_hidden_size, verbose_training=False):
        """Create the LSTM with AutoEncoder-compressed inputs.

        :ae_input_size: Input size to the AutoEncoder, should be n_coins
        :ae_hidden_size: Size of the middle layer in the AutoEncoder
        :lstm_hidden_size: Size of the hidden dimension in the LSTM predictor

        """

        # Arguments
        self.ae_in = ae_input_size
        self.ae_hid = ae_hidden_size
        self.lstm_hid = lstm_hidden_size
        self.verbose_training = verbose_training

        # Torch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        # Hard-coded variables
        self.ae_epochs = 200
        self.ae_learning_rate = 0.001
        self.lstm_epochs = 6
        self.lstm_learning_rate = 0.001

        # Encoder
        self.ae = self.AutoEncoder(self.ae_in, self.ae_hid)
        self.ae_training_loss = np.zeros(self.ae_epochs)

        # Initialize multivariate LSTMs
        self.lstm_arr = [0] * self.ae_in
        for i in range(len(self.lstm_arr)):
            self.lstm_arr[i] = SimpleLSTM(self.ae_hid, self.lstm_hid, 1)
        self.lstm_training_loss = np.zeros(self.lstm_epochs)

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: [batch_size, 1, n_coins] Tensor of predictions
        """

        # Set the model to evaluation mode
        #  self.lstm.eval()
        #  with torch.no_grad():
        #      transformed = self.ae.encoder(sample)
        #      return self.lstm(transformed)

        transformed = self.ae.encoder(sample)
        transformed = transformed.clone().float().to(self.device)

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

            # Verbose Debugging
            if self.verbose_training and epoch % (self.ae_epochs // 3) == 0:
                print(f'{self.get_fullname()} at AutoEncoder training epoch {epoch}')

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

        ######## Configure the optimisers ########
        optimisers = [0] * self.ae_in
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
                data = self.ae.encoder(data.clone().float().to(self.device))
                transformed = data.clone().float().to(self.device)
                target = target.clone().float().to(self.device)

                # Perform the training steps for each of the models
                for i in range(len(self.lstm_arr)):
                    t = target[:,:,i].reshape(-1, 1, 1)
                    output = self.lstm_arr[i](transformed)  # Step ①
                    loss = self.criterion(output, t)        # Step ②
                    optimisers[i].zero_grad()               # Step ③
                    loss.backward(retain_graph=True)        # Step ④
                    optimisers[i].step()                    # Step ⑤

            self.lstm_training_loss[epoch] = loss.item()

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        return f"AE({self.ae_hid},{self.ae_epochs})-LSTM({self.lstm_hid},{self.lstm_epochs})"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        return f"{self.ae_in}Coins-AE({self.ae_hid})_LSTM({self.lstm_hid})"

    def needs_retraining(self):
        """Does this model need regular retraining while forecasting?
        :returns: bool
        """
        return True

    def get_plotting_color(self):
        """return color for graphing distinction
        :returns: str of color
        """
        return "#5a60bd"
