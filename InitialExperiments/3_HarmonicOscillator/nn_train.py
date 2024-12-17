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

def main(N_train_snake, num_epochs, d, w0):
    # Hyperparameters
    batch_size = 25
    input_size, hidden_sizes, output_size = 1, [64, 64], 1
    print_every = int(num_epochs / 10)
    lr = 0.01

    # Parameters for sine wave dataset
    noise_std = 0
    n_waves = 3
    mu, k = 2 * d, w0**2

    # Generate data for snake
    train_x_array_snake, train_y_array_snake, test_x_array, test_y_array = generate_data(N_train_snake, noise_std, n_waves, shifted=True, d=d, w0=w0)
    train_x_snake, train_y_snake, test_x, test_y = func.convert_to_tensor(train_x_array_snake, train_y_array_snake, test_x_array, test_y_array)
    train_loader_snake = func.create_dataloader(train_x_snake, train_y_snake, batch_size=batch_size)

    # SnakeMLP
    SnakeMLP = func.SnakeMLP(input_size, hidden_sizes, output_size)
    model_SnakeMLP, _, _ = func.train_model(SnakeMLP, train_loader_snake, test_x, test_y, lr, num_epochs=num_epochs, print_every=print_every)
    model_SnakeMLP.eval()

    with torch.no_grad():
        model_SnakeMLP.eval()
        test_pred_snake = model_SnakeMLP(test_x).numpy().squeeze()

    # Print mse on test data
    mse_test = np.mean((test_y_array - test_pred_snake)**2)
    print("MSE on test data: ", mse_test)

    ## plot test data
    #plt.figure(figsize=(12, 6))
    #plt.plot(test_x, test_y, label="True")
    #plt.plot(test_x, test_pred_snake, label="Predicted")
    #plt.legend()
    #plt.show()
    #plt.close()

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train SnakeMLP with customizable parameters.")
    parser.add_argument("--N_train_snake", type=int, default=4000, help="Number of training points for SnakeMLP")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--d", type=float, default=0.0, help="Damping coefficient for the underdamped harmonic oscillator")
    parser.add_argument("--w0", type=float, default=1.0, help="Angular frequency for the underdamped harmonic oscillator")
    args = parser.parse_args()
    
    main(args.N_train_snake, args.num_epochs, args.d, args.w0)