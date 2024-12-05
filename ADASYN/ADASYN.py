import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN, RandomOverSampler


data = pd.read_csv("merged_file.csv")


data['marker'] = pd.to_numeric(data['marker'], errors='coerce')
data['marker'] = data['marker'].apply(lambda x: 1 if x >= 7 and x <= 12 else 0)


X = data.drop(columns=['marker'])
y = data['marker']


X = X.replace([np.inf, -np.inf], np.nan)


X = X.dropna()
y = y[X.index]


rmt = RandomOverSampler(sampling_strategy='not minority', random_state=42)
X_rmt, y_rmt = rmt.fit_resample(X, y)


adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_rmt, y_rmt)


original_samples = len(y_rmt)
synthetic_samples = len(y_resampled) - original_samples


X_synthetic = X_resampled[original_samples:]
y_synthetic = y_resampled[original_samples:]


synthetic_data = pd.concat([X_synthetic, y_synthetic], axis=1)


synthetic_data.to_csv('synthetic_data.csv', index=False)