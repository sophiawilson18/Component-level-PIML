#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from math import *
import sys

import scipy
from scipy import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Color maps
cmap = plt.get_cmap('inferno')
cmap2 = plt.get_cmap('seismic')

# ADJUSTABLE PARAMETERS
layers = [2] + [50] * 4 + [1]
epochs = 9000
N_train = 4456  # Number of training samples
N_val = 0    # Number of validation samples

def main():
    # Define domain
    nx = 256                     # spatial resolution
    nt = 100                     # temporal resolution
    x = np.linspace(-1, 1, nx)   # spatial grid
    t = np.linspace(0, 1, nt)    # temporal grid
    X, T = np.meshgrid(x, t)     # 2D grid for the domain

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # All (x, t) coordinate pairs
    print(f"X_star shape: {X_star.shape}")

    # Load the data
    u_star_array = np.load('burgers_sol.npy', allow_pickle=True) 
    u_star = u_star_array[-1]

    # Define indices for training, validation, and testing points
    idx_all = np.arange(X_star.shape[0])  # All indices
    np.random.shuffle(idx_all)            # Shuffle indices

    idx_train = idx_all[:N_train]        # First N_train points for training
    idx_val = idx_all[N_train:N_train+N_val]  # Next N_val points for validation

    # Separate training, validation, and testing data
    X_train = X_star[idx_train]
    y_train = u_star[idx_train].squeeze()

    X_val = X_star[idx_val]
    y_val = u_star[idx_val].squeeze()

    # Prepare data
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)


    # Define neural network
    class NeuralNetwork(nn.Module):
        def __init__(self, layers):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(len(layers) - 1):
                layer = nn.Linear(layers[i], layers[i+1])
                nn.init.xavier_normal_(layer.weight)  # Xavier initialization
                nn.init.zeros_(layer.bias)  # Zero bias
                self.layers.append(layer)
            self.activation = torch.tanh

        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                x = self.activation(layer(x))
            x = self.layers[-1](x)  # No activation for the last layer
            return x

    # Initialize model
    model = NeuralNetwork(layers)

    # Set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Track losses
    train_losses = []
    val_losses = []

    # Train the model
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = criterion(y_val_pred, y_val_tensor).item()
        val_losses.append(val_loss)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Training Loss: {epoch_train_loss:.6e}, Validation Loss: {val_loss:.6e}')

    # Predict on the entire domain
    X_star_tensor = torch.tensor(X_star, dtype=torch.float32)
    y_test_pred = model(X_star_tensor).detach().numpy()

    # MSE 
    testing_error = np.mean((y_test_pred.flatten() - u_star.flatten())**2)
    print(f"Testing MSE by Neural Network: {testing_error:.6e}")

    ## Plot training and validation loss
    #plt.figure(figsize=(10, 6))
    #plt.plot(range(epochs), train_losses, label='Training Loss')
    #plt.plot(range(epochs), val_losses, label='Validation Loss')
    #plt.xlabel('Epochs', fontsize=12)
    #plt.ylabel('Loss', fontsize=12)
    #plt.title('Training and Validation Loss', fontsize=14)
    #plt.legend()
    #plt.grid(True)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.savefig('Plots/burgers_nn_loss.png')
    #plt.show()
    #plt.close()
#
    ## Visualization
    #fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
#
    ## Reference solution
    #scatter_ref = axes[0].scatter(
    #    X_star[:, 0], X_star[:, 1], c=u_star, cmap=cmap, vmin=-1, vmax=1, s=5
    #)
    #axes[0].set_title('Target', fontsize=15)
    #axes[0].set_xlabel('x', fontsize=12)
    #axes[0].set_ylabel('t', fontsize=12)
    #plt.colorbar(scatter_ref, ax=axes[0])
#
    ## Neural network prediction
    #scatter_pred = axes[1].scatter(
    #    X_star[:, 0], X_star[:, 1], c=y_test_pred.flatten(), cmap=cmap, vmin=-1, vmax=1, s=5
    #)
    #axes[1].set_title('Prediction', fontsize=15)
    #axes[1].set_xlabel('x', fontsize=12)
    #axes[1].set_ylabel('t', fontsize=12)
    #plt.colorbar(scatter_pred, ax=axes[1])
#
    ## Error plot
    #c_lim = max(np.abs(u_star.flatten() - y_test_pred.flatten()))
    #scatter_error = axes[2].scatter(
    #    X_star[:, 0], X_star[:, 1], c=np.abs(u_star.flatten() - y_test_pred.flatten()), cmap=cmap2, vmin=-c_lim, vmax=c_lim, s=5
    #)
    #axes[2].set_title('Prediction Error', fontsize=15)
    #axes[2].set_xlabel('x', fontsize=12)
    #axes[2].set_ylabel('t', fontsize=12)
    #plt.colorbar(scatter_error, ax=axes[2])
#
    #plt.suptitle("Burgers' Equation: Neural Network Results", fontsize=18)
    #plt.savefig('Plots/burgers_nn_results.png')
    #plt.show()
    #plt.close()
#
if __name__ == "__main__":
    main()