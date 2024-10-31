import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
import numpy as np

data = pd.read_csv("merged_file.csv")
data['marker'] = pd.to_numeric(data['marker'], errors='coerce')
data['marker'] = data['marker'].apply(lambda x: 1 if x >= 7 and x <= 12 else 0)
X = data.drop('marker', axis=1)
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.max().max(), inplace=True)
y = data['marker']

ros = RandomOverSampler(sampling_strategy='not minority')
X_resampled, y_resampled = ros.fit_resample(X, y)
X_resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
X_resampled.fillna(X_resampled.max().max(), inplace=True)

smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

# Separate synthetic samples
n_original = len(y)  # Number of original samples
X_synthetic = X_resampled[n_original:]
y_synthetic = y_resampled[n_original:]

# Combine synthetic samples into a new DataFrame
synthetic_data = pd.concat([X_synthetic, y_synthetic], axis=1)

# Save synthetic data to a new CSV file
synthetic_data.to_csv("only_synthetic_data.csv", index=False)


