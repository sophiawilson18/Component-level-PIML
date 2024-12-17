import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import sys

# Append custom module path
sys.path.append("..//1_Interpolation")
import E1_Functions as func

# Helper Function
def oscillator(d, w0, x):
    """Analytical solution to the 1D underdamped harmonic oscillator."""
    assert d < w0, "Damping coefficient must be less than angular frequency for underdamped motion."
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    y = torch.exp(-d * x) * 2 * A * torch.cos(phi + w * x)
    return y

# Define MLP
class SnakeActivation(nn.Module):
    def forward(self, x):
        return x + torch.sin(x)**2

class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(SnakeActivation())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def main():
    # Parameters
    num_epochs = 100
    num_points = 10000  
    num_physics = 100
    

    # Parameters for sine wave dataset
    n_waves = 5.5  # 5.5
    noise_std = 0.3
    d, w0 = 0, 1
    mu, k = 2 * d, w0**2    

    # Initialize Model
    hidden_sizes = [64, 64] #[32, 32, 32]
    batch_size = 25
    model = MLP(input_size=1, hidden_sizes=hidden_sizes, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Generate data
    x = torch.linspace(-n_waves * np.pi, n_waves * np.pi, num_points)
    x_physics = torch.linspace(-n_waves * np.pi, 3 * n_waves * np.pi, num_physics).requires_grad_(True)
    y = oscillator(d, w0, x)
    x_train = x.unsqueeze(1)
    y_train = y.unsqueeze(1) + noise_std * torch.randn_like(y).unsqueeze(1)

    test_x = np.linspace(n_waves * np.pi, 3 * n_waves * np.pi, 100)        
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)  # Convert to tensor and match shape
    test_y = np.cos(test_x) + np.random.normal(0, noise_std, 100)
    test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1)  # Convert to tensor and match shape


    # Data Loader
    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    # Training Loop
    with tqdm.tqdm(total=num_epochs) as pbar:
        for step in range(num_epochs):
            model.train()
            for inputs, targets in train_dataloader:
                # Data Loss
                outputs_data = model(inputs)
                loss_data = criterion(outputs_data, targets)
                
                # Physics Loss
                outputs_physics = model(x_physics.unsqueeze(1))
                dx = torch.autograd.grad(outputs_physics, x_physics, torch.ones_like(outputs_physics),
                                        create_graph=True, retain_graph=True)[0]
                dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0]
                physics = dx2 + mu * dx + k * outputs_physics
                loss_physics = 1e-4 * torch.mean(physics**2)
                
                # Total Loss
                loss = loss_data + loss_physics
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update Progress Bar
            pbar.update(1)


    with torch.no_grad():
                model.eval()
                y_pred = model(test_x_tensor).detach()

    # Print MSE test loss
    criterion = nn.MSELoss()
    mse = criterion(y_pred, test_y)
    print(f"MSE Test Loss: {mse.item()}")

    # plot analytical solution on top of training data
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x_train.detach(), y_train.detach(), label='Train data', color='grey', marker='o', ms=3, ls='-')
    ax.plot(x, y, label='Analytical solution', color='black')
    ax.plot(test_x, test_y, label='Test data', color='black', marker='o', ms=3, ls='-')
    ax.plot(test_x, y_pred, label='Neural network')
    ax.legend()
    plt.savefig("results_pinn.png")
    plt.close()


if __name__ == "__main__":
    main()

    
