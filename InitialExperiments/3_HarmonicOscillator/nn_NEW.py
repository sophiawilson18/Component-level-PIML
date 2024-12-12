import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm
import sys

# Append custom module path
sys.path.append("..//Experiment1")
import E1_Functions as func


def main():
    # Parameters
    num_points = 10000
    num_epochs = 100

    # Parameters for sine wave dataset
    n_waves = 5.5  # 5.5
    noise_std = 0.3

    # Parameters for neural network
    input_size, hidden_sizes, output_size = 1, [64, 64], 1
    batch_size = 25
    print_every = int(num_epochs / 10)
    lr = 0.01

    # Generate data
    train_x_array, train_y_array, test_x_array, test_y_array = func.generate_data(
        num_points, noise_std, n_waves, shifted=True
    )
    train_x, train_y, test_x, test_y = func.convert_to_tensor(
        train_x_array, train_y_array, test_x_array, test_y_array
    )
    train_loader = func.create_dataloader(train_x, train_y, batch_size=batch_size)

    # Train model
    SnakeMLP = func.SnakeMLP(input_size, hidden_sizes, output_size)
    model_SnakeMLP, _, _ = func.train_model(
        SnakeMLP, train_loader, test_x, test_y, lr, num_epochs=num_epochs, print_every=print_every
    )

    model_SnakeMLP.eval()
    with torch.no_grad():
        pred_y = model_SnakeMLP(test_x).numpy().squeeze()

    # Print MSE test loss
    criterion = nn.MSELoss()
    mse = criterion(torch.tensor(pred_y).view_as(test_y), test_y)
    print(f"MSE Test Loss: {mse.item()}")

    # Plot results
    plt.figure()
    plt.plot(test_x_array, test_y_array, label="Exact solution")
    plt.plot(test_x_array, pred_y, color="black", label="NN prediction")
    plt.legend()
    plt.savefig("results_nn.png")
    plt.close()



if __name__ == "__main__":
    main()