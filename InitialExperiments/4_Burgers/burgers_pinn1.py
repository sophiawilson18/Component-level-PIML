#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys

# Import PINNs module
sys.path.append('../')  # Ensure module path is accessible
import importlib
import pinns
importlib.reload(pinns)
from pinns import PINNs

# Color maps
cmap = plt.get_cmap('inferno')
cmap2 = plt.get_cmap('seismic')

# ADJUSTABLE PARAMETERS
layers = [2] + [50] * 4 + [1]  # Depth and width of the neural network
lr = 0.001  # Learning rate
N_f = 1000  # Number of collocation points
nu = 0.025  # PDE parameter (viscosity)
adam_epochs = 2000
lbfgs_epochs = 3000

def main():
    # Define the domain
    nx = 256                     # Spatial resolution
    nt = 100                     # Temporal resolution
    x = np.linspace(-1, 1, nx)   # Spatial grid
    t = np.linspace(0, 1, nt)    # Temporal grid
    X, T = np.meshgrid(x, t)     # 2D grid for the domain

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    print(f"X_star shape: {X_star.shape}")

    # Initial condition (IC)
    idx_init = np.where(X_star[:, 1] == 0)[0]
    X_init = X_star[idx_init]
    u_init = -np.sin(np.pi * X_init[:, 0:1]).squeeze()

    # Boundary conditions (BC)
    idx_bc = np.where((X_star[:, 0] == 1.0) | (X_star[:, 0] == -1.0))[0]
    X_bc = X_star[idx_bc]
    u_bc = np.zeros((X_bc.shape[0], 1)).squeeze()

    # Collocation points
    idx_Xf = np.random.choice(X_star.shape[0], N_f, replace=False)
    X_colloc = X_star[idx_Xf]

    # Convert data to PyTorch tensors
    X_colloc_train_tensor = torch.tensor(X_colloc, dtype=torch.float64)
    X_bc_tensor = torch.tensor(X_bc, dtype=torch.float64)
    u_bc_tensor = torch.tensor(u_bc, dtype=torch.float64).view(-1, 1)
    X_init_tensor = torch.tensor(X_init, dtype=torch.float64)
    u_init_tensor = torch.tensor(u_init, dtype=torch.float64).view(-1, 1)

    # Define neural network transformation
    def net_transform(X_f, model_nn):
        return model_nn(X_f)

    def f_user(X_f, model_nn, nu):
        x_temp = X_f[:, 0:1].clone().detach()
        t_temp = X_f[:, 1:2].clone().detach()

        # Enable gradient tracking
        x_temp.requires_grad_(True)
        t_temp.requires_grad_(True)

        # Concatenate inputs
        X_temp = torch.cat([x_temp, t_temp], dim=1)

        # Compute network output
        u = net_transform(X_temp, model_nn)

        # Compute derivatives using PyTorch autograd
        u_x = torch.autograd.grad(u, x_temp, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_temp, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_temp, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute PDE residual
        f = u_t + u * u_x - nu * u_xx
        return f

    # Define residual loss
    def loss_f(f):
        return torch.mean(f ** 2)


    # Instantiate the model
    model = PINNs(
        X_colloc_train_tensor,
        net_transform,
        lambda X, model: f_user(X, model, nu),
        loss_f,
        layers,
        lr,
        type_problem='forward',
        X_bc=X_bc_tensor,
        u_bc=u_bc_tensor,
        X_init=X_init_tensor,
        u_init=u_init_tensor,
    )

    model.net_u = model.net_u.double()


    # Train the model
    print("Training the PINN...")
    model.train(max_epochs_adam=adam_epochs, max_epochs_lbfgs=lbfgs_epochs, print_per_epochs=100)


    # Predict on the entire domain
    X_star_tensor = torch.tensor(X_star, dtype=torch.float64)
    with torch.no_grad():
        pred = model.net_transform(X_star_tensor, model.net_u).numpy()

    # Load the reference solution
    u_star_array = np.load('burgers_sol.npy', allow_pickle=True)
    u_star = u_star_array[-1]

    # MSE 
    mse = np.mean((u_star.flatten() - pred.flatten()) ** 2)
    print(f"(Testing MSE by PINN) = {mse:.6e}")    


    # Visualization
    fig = plt.figure(figsize=[15, 4])
    axes = fig.subplots(1, 3, sharex=False, sharey=False)

    # Reference solution
    img1 = axes[0].scatter(
        X_star[:, 0:1], X_star[:, 1:2], c=u_star, cmap=cmap, vmax=1, vmin=-1, s=5
    )
    axes[0].set_title('Reference Solution', fontsize=15)
    axes[0].set_xlabel('x', fontsize=15)
    axes[0].set_ylabel('t', fontsize=15)
    plt.colorbar(img1, ax=axes[0])

    # PINNs prediction
    img2 = axes[1].scatter(
        X_star[:, 0:1], X_star[:, 1:2], c=pred, cmap=cmap, vmax=1, vmin=-1, s=5
    )
    axes[1].set_title('PINNs Prediction', fontsize=15)
    axes[1].set_xlabel('x', fontsize=15)
    axes[1].set_ylabel('t', fontsize=15)
    plt.colorbar(img2, ax=axes[1])

    # Error
    error = u_star - pred
    c_lim = max(np.abs(error.flatten()))
    print(f"Error limit: {c_lim}")
    img3 = axes[2].scatter(
        X_star[:, 0:1], X_star[:, 1:2], c=error, cmap='seismic', vmax=c_lim, vmin=-c_lim, s=5
    )
    axes[2].set_title('Error', fontsize=15)
    axes[2].set_xlabel('x', fontsize=15)
    axes[2].set_ylabel('t', fontsize=15)
    plt.colorbar(img3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('Plots/burgers_pinn_results.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()