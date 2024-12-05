import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from xgboost import XGBRegressor



# Load the original dataset
data = pd.read_csv("IEEE118NormalWithPd_Qd.csv")

# Select features and target variables
# Independent variables
vgm_columns = [col for col in data.columns if 'VGM' in col]
pg_columns = [col for col in data.columns if 'PG' in col]
pl_columns = [col for col in data.columns if 'PL' in col]
ql_columns = [col for col in data.columns if 'QL' in col]

# Dependent variables
vlm_columns = [col for col in data.columns if 'VLM' in col]
vla_columns = [col for col in data.columns if 'VLA' in col]
vga_columns = [col for col in data.columns if 'VGA' in col]

X = data[vgm_columns + pg_columns + pl_columns + ql_columns]
y = data[vlm_columns + vla_columns + vga_columns]


# Load the trained model
file = open("xgbmodel.pickle",'rb')
loaded_model = pickle.load(file)

# Function to simulate FDI attack
def simulate_fdi_attack(data):
    # for col in vgm_columns:
    #   data[col] += np.random.uniform(-0.15, 0.15)
    for col in pg_columns:
       data[col] *= np.random.uniform(0.1, 0.12)
    for col in pl_columns:
        data[col] *= np.random.uniform(0.11, 0.20)
    for col in ql_columns:
       data[col] += np.random.uniform(-0.05, 0.05)
    return data

# Simulate FDI attack on the original data
attacked_data = simulate_fdi_attack(X.copy())  # Create a copy to avoid modifying the original data
#attacked_data.to_csv("attack_data.csv",index=None)

# Make predictions on the attacked data
predicted_values = loaded_model.predict(attacked_data)

# Create a new DataFrame with original data and predicted values under attack
result_df = pd.concat([attacked_data, pd.DataFrame(predicted_values, columns=y.columns)], axis=1)

# Save the new dataset
result_df.to_csv('Combine_class.csv', index=False)

