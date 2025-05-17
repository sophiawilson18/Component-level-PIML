"""
create_dataframe.py

This script restructures the data and saves it as a CSV file.
"""

import pandas as pd
import ast

# Load data
df = pd.read_csv('CylinderFlowData.csv')

# Extract number of nodes
N_nodes = int(df['ptr'].iloc[2])

# Cut data to N_nodes
df = df.iloc[:N_nodes]

# Efficiently parse columns with literal_eval
parsed_columns = ['x', 'y', 'pos']
for col in parsed_columns:
    df[col] = df[col].map(ast.literal_eval)

# Extract data from parsed columns
x_data = df['x']
y_data = df['y']
pos = df['pos']

# Calculate time step (dt)
dt = abs(x_data.iloc[0][-1] - x_data.iloc[-1][-1])

# Create dataframe with extracted data
data = pd.DataFrame({
    'node_type': x_data.map(lambda x: x[0] if x else None),   # Node type
    'pos_x': pos.map(lambda x: x[0] if x else None),          # Position x
    'pos_y': pos.map(lambda x: x[1] if len(x) > 1 else None), # Position y
    'vx': y_data.map(lambda x: x[0] if x else None),          # Velocity x
    'vy': y_data.map(lambda x: x[1] if len(x) > 1 else None), # Velocity y
    'p': x_data.map(lambda x: x[3] if len(x) > 3 else None),  # Pressure
    't': x_data.map(lambda x: x[4] if len(x) > 4 else None),  # Time
})

# Split data into two time steps (obs: t2 is the first half of the data)
mid_idx = N_nodes // 2
data_t1 = data.iloc[mid_idx:].reset_index(drop=True)
data_t2 = data.iloc[:mid_idx].reset_index(drop=True)

# Save the DataFrame to a CSV file
output_path = "cylinderflow_dataframe.csv"
data_t1.to_csv(output_path, index=False)
print(f"DataFrame saved to {output_path}")
