import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
from importlib import reload
import argparse

import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Append the path to import custom functions
sys.path.append("..//1_Interpolation")
import E1_Functions as func

# Helper Function: Analytical solution to the 1D underdamped harmonic oscillator
def oscillator(d, w0, x):
    """Analytical solution to the 1D underdamped harmonic oscillator."""
    assert d < w0, "Damping coefficient must be less than angular frequency for underdamped motion."
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    y = torch.exp(-d * x) * 2 * A * torch.cos(phi + w * x)
    return y

# Updated generate_data function
def generate_data(num_points, noise_std=0.3, n_waves=15, shifted=False, d=0, w0=1):
    # Generate train_x in the specified range
    train_x = np.linspace(-n_waves * np.pi, n_waves * np.pi, num_points)
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    
    # Generate train_y using the oscillator function and add noise
    train_y = oscillator(d, w0, train_x_tensor).numpy() + np.random.normal(0, noise_std, num_points)

    if shifted:  # For shifted test data
        test_x = np.linspace(n_waves * np.pi, 3 * n_waves * np.pi, 100)
    else:        # For non-shifted test data
        test_x = np.linspace(-n_waves * np.pi, n_waves * np.pi, 100)

    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y = oscillator(d, w0, test_x_tensor).numpy()

    return train_x, train_y, test_x, test_y

def main(N_train_pinn, N_physics, num_epochs, d, w0):
    # Hyperparameters
    batch_size = 25
    input_size, hidden_sizes, output_size = 1, [64, 64], 1
    print_every = 300
    lr = 0.01

    # Parameters for sine wave dataset
    noise_std = 0
    n_waves = 3
    mu, k = 2 * d, w0**2

    # Generate data for pinn
    train_x_array_pinn, train_y_array_pinn, test_x_array, test_y_array = generate_data(N_train_pinn, noise_std, n_waves, shifted=True, d=d, w0=w0)
    train_x_pinn, train_y_pinn, test_x, test_y = func.convert_to_tensor(train_x_array_pinn, train_y_array_pinn, test_x_array, test_y_array)
    train_loader_pinn = func.create_dataloader(train_x_pinn, train_y_pinn, batch_size=batch_size)
    x_physics = torch.linspace(-n_waves * np.pi, 3 * n_waves * np.pi, N_physics).requires_grad_(True).view(-1, 1)

    # PINN
    model = func.SnakeMLP(input_size, hidden_sizes, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    
    with tqdm(total=num_epochs) as pbar:
        for step in range(num_epochs):
            model.train()
            for inputs, targets in train_loader_pinn:
                # Data Loss
                outputs_data = model(inputs)
                loss_data = criterion(outputs_data, targets)
                # Physics Loss
                outputs_physics = model(x_physics)
                dx = torch.autograd.grad(outputs_physics, x_physics, torch.ones_like(outputs_physics), create_graph=True)[0]
                dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]
                physics = dx2 + mu * dx + k * outputs_physics
                loss_physics =  torch.mean(physics**2)
                
                # Total Loss
                loss = loss_physics + loss_data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update Progress Bar
            pbar.update(1)

    # Evaluation
    with torch.no_grad():
        model.eval()
        test_pred_pinn = model(test_x).detach().numpy()

    # Print mse on test data
    mse_test = np.mean((test_y_array - test_pred_pinn.squeeze())**2)
    print("MSE on test data: ", mse_test)

    # plot test data
    #plt.figure(figsize=(12, 6))
    #plt.plot(test_x, test_y, label="True")
    #plt.plot(test_x, test_pred_pinn, label="Predicted")
    #plt.legend()
    #plt.show()
    #plt.close()
#
if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train PINN with customizable parameters.")
    parser.add_argument("--N_train_pinn", type=int, default=50, help="Number of training points for PINN")
    parser.add_argument("--N_physics", type=int, default=50, help="Number of points for the physics-informed loss")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of epochs for training")
    parser.add_argument("--d", type=float, default=0.0, help="Damping coefficient for the underdamped harmonic oscillator")
    parser.add_argument("--w0", type=float, default=1.0, help="Angular frequency for the underdamped harmonic oscillator")
    args = parser.parse_args()
    
    main(args.N_train_pinn, args.N_physics, args.num_epochs, args.d, args.w0)