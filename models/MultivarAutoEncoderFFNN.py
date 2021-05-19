#!/usr/bin/env python

# Standard imports
import pandas as pd
import numpy as np

# Pytorch
import torch
from torch import nn

# Local Files
from models.model_interface import CryptoModel

class MultivarAutoEncoderFFNN(CryptoModel):
    """1-Layer Feed-Forward Neural Network with inputs transformed by a 1-layer AutoEncoder"""

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

    def __init__(self, ae_input_size, ae_hidden_size, nn_hidden_size, verbose_training=False):
        """Create the FFNN with AutoEncoder-compressed inputs.

        :ae_input_size: Input size to the AutoEncoder, should be n_coins
        :ae_hidden_size: Size of the middle layer in the AutoEncoder
        :nn_hidden_size: Size of the hidden dimension in the FFNN predictor

        """

        # Arguments
        self.ae_in = ae_input_size
        self.ae_hid = ae_hidden_size
        self.nn_hid = nn_hidden_size
        self.verbose_training = verbose_training

        # Torch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MSELoss()

        # Hard-coded variables
        self.ae_epochs = 200
        self.ae_learning_rate = 0.001
        self.nn_epochs = 6
        self.nn_learning_rate = 0.001

        # Encoder
        self.ae = self.AutoEncoder(self.ae_in, self.ae_hid)
        self.ae_training_loss = np.zeros(self.ae_epochs)

        # Initialize multivariate FFNNs
        self.nn_arr = [0] * self.ae_in
        for i in range(len(self.nn_arr)):
            self.nn_arr[i] = nn.Sequential(
                nn.Linear(self.ae_hid, self.nn_hid),
                nn.Linear(self.nn_hid, 1)
            )
        self.nn_training_loss = np.zeros(self.nn_epochs)

        self.set_plotting_color()

    def predict(self, sample):
        """Predict the next out of sample timestep
        :sample: Vector or DataFrame of timesteps to use as input for the predictor(s).
        :returns: [batch_size, 1, n_coins] Tensor of predictions
        """

        transformed = self.ae.encoder(sample)
        transformed = transformed.clone().float().to(self.device)

        ret_list = []
        for i in range(len(self.nn_arr)):
            # Set the model to evaluation mode
            self.nn_arr[i].eval()
            with torch.no_grad():
                ret = self.nn_arr[i](transformed)
                ret_list.append(ret)

        return torch.cat(ret_list, 2)

    def train(self, training_set):
        """Train, or re-train, the FFNN and AE

        :training_set: DataFrame of training samples
        """
        self._train_ae(training_set)
        self._train_nn(training_set)

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

    def _train_nn(self, training_set):
        """Helper method to contain the training cycle for the FFNN
        :training_set: PyTorch dataloader for training
        """

        ######## Configure the optimisers ########
        optimisers = [0] * self.ae_in
        for i in range(len(self.nn_arr)):
            optimisers[i] = torch.optim.Adam(
                self.nn_arr[i].parameters(),
                lr=self.nn_learning_rate,
            )

            # Set the FFNNs to training mode
            self.nn_arr[i].train()

        # Iterate over every batch of sequences
        for epoch in range(self.nn_epochs):

            # Verbose Debugging
            if self.verbose_training and (epoch % (self.nn_epochs // 3) == 0):
                print(f'{self.get_fullname()} at FFNN training epoch {epoch}')

            for data, target in training_set:

                # reshape target per pytorch warnings
                t_shape = target.shape
                target = target.reshape(t_shape[0], 1, t_shape[1])

                # Convert
                data = self.ae.encoder(data.clone().float().to(self.device))
                transformed = data.clone().float().to(self.device)
                target = target.clone().float().to(self.device)

                # Perform the training steps for each of the models
                for i in range(len(self.nn_arr)):
                    t = target[:,:,i].reshape(-1, 1, 1)
                    output = self.nn_arr[i](transformed)  # Step ①
                    loss = self.criterion(output, t)        # Step ②
                    optimisers[i].zero_grad()               # Step ③
                    loss.backward(retain_graph=True)        # Step ④
                    optimisers[i].step()                    # Step ⑤

            self.nn_training_loss[epoch] = loss.item()

    def get_fullname(self):
        """Get the full-grammar name for this model
        :returns: English phrase as string
        """
        return f"AE({self.ae_hid},{self.ae_epochs})-FFNN({self.nn_hid},{self.nn_epochs})"

    def get_filename(self):
        """Get the abbreviated (file)name for this model
        :returns: Abbreviated string with underscores
        """
        return f"{self.ae_in}Coins-AE({self.ae_hid})_FFNN({self.nn_hid})"

    def needs_retraining(self):
        """Does this model need regular retraining while forecasting?
        :returns: bool
        """
        return True

    def set_plotting_color(self, color="#5a60bd"):
        """Set the color used for plotting model outcomes
        :color: String containing hex value for color
        """
        self.color = color

    def get_plotting_color(self):
        """return color for graphing distinction
        :returns: str of color
        """
        return self.color
