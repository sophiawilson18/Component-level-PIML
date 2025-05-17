"""
train_model.py

This script trains the model using the dataframe and saves the plots and parameters.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
from datetime import datetime
import seaborn as sns
import pandas as pd
from datetime import datetime
import os

import sys
sys.path.append('../')  
import importlib
import pinns  
importlib.reload(pinns)
from pinns import PINNs

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define a discrete color palette with a specified number of colors
#sns.set_palette(sns.color_palette("muted", n_colors=10))  # or use another color map and adjust `n_colors`
custom_palette = ["#FFB347", "#779ECB", "#77DD77", "#FF6961", "royalblue", "#CB99C9", "#FDFD96", "#AEC6CF"]
costum_palette_pairs = ['#ffb347', '#ffcc84', '#ffe6c2', 
                        '#779ecb', '#a4bedc', '#d2dfee', 
                        '#77dd77', '#a4e8a4', '#d2f4d2', 
                        '#ff6961', '#ff9b96', '#ffcdca', 
                        '#cb99c9', '#dcbbdb', '#eedded', 
                        '#fdfd96', '#fefeb9', '#fefedc',
                        '#aec6cf', '#c9d9df', '#e4ecef']

sns.set_palette(custom_palette)
colors = sns.color_palette()
colors_pairs = sns.color_palette(costum_palette_pairs) #("tab20c")

cmap1 = plt.get_cmap('inferno')
cmap2 = plt.get_cmap('viridis')
cmap3 = plt.get_cmap('binary') #('seismic')

# set title font size
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ============================
# Argument Parser for Parameters
# ============================

parser = argparse.ArgumentParser(description="Train PINN Model on CylinderFlow Data")
parser.add_argument("--n_physics", type=int, default=200, help="Number of training collocation points")
parser.add_argument("--n_supervised", type=int, default=200, help="Number of supervised data points")
parser.add_argument("--n_val", type=int, default=100, help="Number of validation points")
parser.add_argument("--n_hidden_layers", type=int, default=4, help="Number of hidden layers in the neural network")
parser.add_argument("--n_hidden_neurons", type=int, default=64, help="Number of hidden neurons in the neural network")
parser.add_argument("--epochs_adam", type=int, default=0, help="Number of epochs for Adam optimizer")
parser.add_argument("--epochs_lbfgs", type=int, default=2000, help="Number of epochs for L-BFGS optimizer")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate for Adam optimizer")
parser.add_argument("--Re_number", type=float, default=3000.0, help="Reynolds number for the flow")
parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plots and parameters")


args = parser.parse_args()

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# ============================
# Load DataFrame
# ============================

print("Loading data...")
data_path = "cylinderflow_dataframe.csv"
data = pd.read_csv(data_path)
print(f"Data loaded from {data_path}")

# ============================
# Prepare Data
# ============================

print("Preparing data...")
X_star = torch.tensor(data[['pos_x', 'pos_y', 't']].values, dtype=torch.float64, requires_grad=True)
y_star = torch.tensor(data[['vx', 'vy', 'p']].values, dtype=torch.float64, requires_grad=True)

# Ensure no overlap between datasets
total_points = X_star.shape[0]
assert args.n_physics + args.n_supervised + args.n_val <= total_points, "Dataset sizes exceed total points"

# Shuffle indices
all_indices = np.arange(total_points)
np.random.shuffle(all_indices)

# Split indices
idx_physics = all_indices[:args.n_physics]
idx_supervised = all_indices[args.n_physics:args.n_physics + args.n_supervised]
idx_val = all_indices[args.n_physics + args.n_supervised:args.n_physics + args.n_supervised + args.n_val]

# Inlet and outlet data (supervised for a unique solution) for node_type == 4.0 and node_type == 5.0
X_boundary = torch.tensor(data[data['node_type'].isin([4.0, 5.0])][['pos_x', 'pos_y', 't']].values, dtype=torch.float64, requires_grad=True)
y_boundary = torch.tensor(data[data['node_type'].isin([4.0, 5.0])][['vx', 'vy', 'p']].values, dtype=torch.float64)

# Create datasets
X_colloc = X_star[idx_physics]
y_colloc = y_star[idx_physics]
X_supervised = X_star[idx_supervised]
y_supervised = y_star[idx_supervised]
X_val = X_star[idx_val]
y_val = y_star[idx_val]

print(f"Training data shape: {X_colloc.shape}")
print(f"Supervised data shape: {X_supervised.shape}")
print(f"Validation data shape: {X_val.shape}")

# ============================
# Plot Data
# ============================

# Plot training data
fig = plt.figure(figsize=(10, 3))
plt.scatter(X_colloc[:, 0].detach().numpy(), X_colloc[:, 1].detach().numpy(), s=5, c='grey', label='Collocation points')
plt.scatter(X_boundary[:, 0].detach().numpy(), X_boundary[:, 1].detach().numpy(), s=5, c=y_boundary[:, -1].detach().numpy(), cmap='viridis', label='Inlet points')
plt.scatter(X_supervised[:, 0].detach().numpy(), X_supervised[:, 1].detach().numpy(), s=5, c=y_supervised[:, -1].detach().numpy(), cmap='viridis', label='Supervised points')
plt.title('Training data')

plt.legend(loc='upper right', frameon=True)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.tight_layout()
plot_path = os.path.join(args.output_dir, 'training_data.png')



# ============================
# Define PDEs / Loss Function
# ============================

print("Defining PDEs and loss function...")

def net_transform(X_f, model_nn):
    return model_nn(X_f)


def f_user(X_f, model_nn, Re=args.Re_number):
    # Extract x, y, t from the input tensor
    x_temp = X_f[:, 0:1]
    y_temp = X_f[:, 1:2]
    t_temp = X_f[:, 2:3]

    # Ensure gradients are enabled for these variables
    x_temp.requires_grad = True
    y_temp.requires_grad = True
    t_temp.requires_grad = True


    # Model predictions
    X_temp = torch.cat([x_temp, y_temp, t_temp], dim=1)
    pred = model_nn(X_temp)
    
    # Outputs: momentum components (w_x, w_y) and pressure (p)
    w_x = pred[:, 0:1]
    w_y = pred[:, 1:2]
    p = pred[:, 2:3]

    # First-order derivatives
    w_x_t = torch.autograd.grad(w_x, t_temp, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_y_t = torch.autograd.grad(w_y, t_temp, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]

    w_x_x = torch.autograd.grad(w_x, x_temp, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_y_x = torch.autograd.grad(w_y, x_temp, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x_temp, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    w_x_y = torch.autograd.grad(w_x, y_temp, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_y_y = torch.autograd.grad(w_y, y_temp, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_temp, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Second-order derivatives
    w_x_xx = torch.autograd.grad(w_x_x, x_temp, grad_outputs=torch.ones_like(w_x_x), create_graph=True)[0]
    w_y_xx = torch.autograd.grad(w_y_x, x_temp, grad_outputs=torch.ones_like(w_y_x), create_graph=True)[0]

    w_x_yy = torch.autograd.grad(w_x_y, y_temp, grad_outputs=torch.ones_like(w_x_y), create_graph=True)[0]
    w_y_yy = torch.autograd.grad(w_y_y, y_temp, grad_outputs=torch.ones_like(w_y_y), create_graph=True)[0]

    # Navier-Stokes momentum equations
    f_1 = w_x_t + w_x * w_x_x + w_y * w_x_y - 1.0 / Re * (w_x_xx + w_x_yy) + p_x
    f_2 = w_y_t + w_x * w_y_x + w_y * w_y_y - 1.0 / Re * (w_y_xx + w_y_yy) + p_y

    # Continuity equation (incompressibility constraint)
    f_3 = w_x_x + w_y_y

    return f_1, f_2, f_3

def loss_f(f, weights=[1.0, 1.0, 1.0]):
    return weights[0] * torch.mean(f[0]**2) + weights[1] * torch.mean(f[1]**2) + weights[2] * torch.mean(f[2]**2)

# ============================
# Train Model
# ============================

print("Training model...")
layers = [3] + args.n_hidden_layers*[args.n_hidden_neurons] + [3]

model_pinn = PINNs(X_colloc, 
             net_transform, 
             f_user, 
             loss_f,
             layers, 
             args.learning_rate,
             type_problem='forward',
             X_data=X_supervised, 
             u_data=y_supervised, 
             X_bc=X_boundary, 
             u_bc=y_boundary,
             X_test=X_val,
             u_test=y_val)

model_pinn.net_u = model_pinn.net_u.double()
model_pinn.train(max_epochs_adam=args.epochs_adam, max_epochs_lbfgs=args.epochs_lbfgs, print_per_epochs=100)


# ============================
# Save Plots and Parameters
# ============================

with torch.no_grad():  # Disable gradient computation for inference
    pred = model_pinn.net_u(X_star).detach().numpy()

# Plot the training loss
plt.figure(figsize=(7, 4))
plt.plot(model_pinn.loss_array, label='Training Loss', color='blue')
plt.plot(model_pinn.test_array, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.yscale('log')
plt.grid(True)
plot_path = os.path.join(args.output_dir, 'training_loss.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")


# Plot results
# Compute the velocity magnitudes for reference, prediction, and error
velocity_ref = np.sqrt(data['vx'].values**2 + data['vy'].values**2)
velocity_pred = np.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
velocity_error = np.sqrt((pred[:, 0] - data['vx'].values)**2 + (pred[:, 1] - data['vy'].values)**2)

# Create triangulation for positions
triang = tri.Triangulation(data['pos_x'], data['pos_y'])

# Speed values
vmin_speed, vmax_speed = velocity_ref.min(), velocity_ref.max()

# Pressure values
vmin_pres, vmax_pres = data['p'].min(), data['p'].max()

# Error values (min and max for both speed and pressure)
vmin_error = np.min([velocity_error.min(), pred[:, 2].min() - data['p'].min()])
vmax_error = np.max([velocity_error.max(), pred[:, 2].max() - data['p'].max()])

# Create the figure and axes (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(13, 5), sharex=True, sharey=True)

# Plot Velocity Magnitude - Reference
axes[0, 0].set_title('A: Speed - Reference')
contour_ref = axes[0, 0].tricontourf(triang, velocity_ref, levels=14, cmap=cmap1, norm=Normalize(vmin=vmin_speed, vmax=vmax_speed))

# Plot Velocity Magnitude - Prediction
axes[1, 0].set_title('C: Speed - Prediction')
contour_pred = axes[1, 0].tricontourf(triang, velocity_pred, levels=14, cmap=cmap1, norm=Normalize(vmin=vmin_speed, vmax=vmax_speed))

# Plot Velocity Magnitude - Error
axes[2, 0].set_title('E: Speed - Error')
contour_err = axes[2, 0].tricontourf(triang, velocity_error, levels=14, cmap=cmap3, norm=Normalize(vmin=vmin_error, vmax=vmax_error))

# Plot Pressure - Reference
axes[0, 1].set_title('B: Pressure - Reference')
pressure_contour_ref = axes[0, 1].tricontourf(triang, data['p'], levels=14, cmap=cmap2, norm=Normalize(vmin=vmin_pres, vmax=vmax_pres))

# Plot Pressure - Prediction
axes[1, 1].set_title('D: Pressure - Prediction')
pressure_contour_pred = axes[1, 1].tricontourf(triang, pred[:, 2], levels=14, cmap=cmap2, norm=Normalize(vmin=vmin_pres, vmax=vmax_pres))

# Plot Pressure - Error
axes[2, 1].set_title('F: Pressure - Error')
pressure_contour_error = axes[2, 1].tricontourf(triang, pred[:, 2] - data['p'], levels=14, cmap=cmap3, norm=Normalize(vmin=vmin_error, vmax=vmax_error))

# Add labels to the axes
for ax in axes[:, 0]:
    ax.set_ylabel('Position y')
for ax in axes[2, :]:
    ax.set_xlabel('Position x')

# Adjust the figure to make space for the colorbars
fig.subplots_adjust(bottom=0.15)

# Add colorbars below the plots
cbar_ax1 = fig.add_axes([0.08, -0.05, 0.25, 0.02])  # Position for speed colorbar
cbar_ax2 = fig.add_axes([0.38, -0.05, 0.25, 0.02])  # Position for pressure colorbar
cbar_ax3 = fig.add_axes([0.68, -0.05, 0.25, 0.02])  # Position for error colorbar

# Create the colorbars
fig.colorbar(contour_ref, cax=cbar_ax1, orientation='horizontal', label='Speed')
fig.colorbar(contour_err, cax=cbar_ax2, orientation='horizontal', label='Error')
fig.colorbar(pressure_contour_ref, cax=cbar_ax3, orientation='horizontal', label='Pressure')

# Display the plot
plt.tight_layout()

# Save the plot
plot_path = os.path.join(args.output_dir, 'results.png')
plt.savefig(plot_path)

# Print mean squared error
mse = np.mean((pred - data[['vx', 'vy', 'p']].values) ** 2)
print("Mean squared error: ", mse)

# Save parameters
params = {
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "MSE loss": mse,
    "Reynolds number": args.Re_number,
    "Number of training points": args.n_physics,
    "Number of supervised points": args.n_supervised,
    "Number of validation points": args.n_val,
    "Number of hidden layers": args.n_hidden_layers,
    "Number of hidden neurons": args.n_hidden_neurons,
    "Epochs (Adam)": args.epochs_adam,
    "Epochs (L-BFGS)": args.epochs_lbfgs,
    "Learning rate": args.learning_rate
}

params_txt_path = os.path.join(args.output_dir, 'training_params.txt')

with open(params_txt_path, 'w') as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

print(f"Training parameters saved to {params_txt_path}")

params_df = pd.DataFrame([params])
params_csv_path = os.path.join(args.output_dir, 'training_params.csv')
params_df.to_csv(params_csv_path, index=False)
print(f"Training parameters saved to {params_csv_path}")
