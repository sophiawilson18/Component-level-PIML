import os
import subprocess

# List of Reynolds numbers to loop through
reynolds_numbers = [1, 100, 1000, 2000, 3000, 4000]

# Base command for training
base_command = "python train_model.py --n_physics 300 --n_supervised 100 --n_val 0 --epochs_adam 0 --epochs_lbfgs 15000 --learning_rate 0.01"

# Output directory base path
output_base_dir = "CF_results"

# Loop through each Reynolds number
for Re in reynolds_numbers:
    output_dir = os.path.join(output_base_dir, f"results_Re_{Re}")
    command = f"{base_command} --Re_number {Re} --output_dir {output_dir}"
    print(f"Running command: {command}")
    
    # Run the command
    subprocess.run(command, shell=True)