import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def clean_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data

def apply_rmt(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)


    covariance_matrix = np.cov(standardized_data, rowvar=False)


    eigenvalues, _ = np.linalg.eigh(covariance_matrix)


    n, p = data.shape
    lambda_max = (1 + np.sqrt(p / n)) ** 2
    lambda_min = (1 - np.sqrt(p / n)) ** 2

    anomalous_indices = np.where((eigenvalues < lambda_min) | (eigenvalues > lambda_max))[0]
    normal_indices = np.where((eigenvalues >= lambda_min) & (eigenvalues <= lambda_max))[0]


    print("Eigenvalues of the covariance matrix:")
    print(eigenvalues)
    print(f"\nRMT Thresholds: Lambda Min = {lambda_min}, Lambda Max = {lambda_max}")
    print(f"\nAnomalous Eigenvalues: {eigenvalues[anomalous_indices]}")
    print(f"\nNormal Eigenvalues: {eigenvalues[normal_indices]}")

    return eigenvalues, anomalous_indices, normal_indices, lambda_min, lambda_max




def summarize_results(eigenvalues, anomalous_indices, normal_indices, lambda_min, lambda_max):

    total_eigenvalues = len(eigenvalues)
    num_anomalous = len(anomalous_indices)
    num_normal = len(normal_indices)

    percentage_anomalous = (num_anomalous / total_eigenvalues) * 100
    percentage_normal = 100 - percentage_anomalous


    print("\n--- Summary of RMT Analysis ---")
    print(f"Total number of eigenvalues: {total_eigenvalues}")
    print(f"Number of anomalous eigenvalues: {num_anomalous} ({percentage_anomalous:.2f}%)")
    print(f"Number of normal eigenvalues: {num_normal} ({percentage_normal:.2f}%)")
    print(f"Thresholds used: Lambda Min = {lambda_min:.2f}, Lambda Max = {lambda_max:.2f}")

    if percentage_anomalous > 30:
        print("High proportion of anomalous eigenvalues")
    elif percentage_anomalous > 10:
        print("Moderate proportion of anomalous eigenvalues")
    else:
        print("Low proportion of anomalous eigenvalues")

    largest_eigenvalue = eigenvalues[-1]
    smallest_eigenvalue = eigenvalues[0]
    print(f"\nThe largest eigenvalue is {largest_eigenvalue:.2f}")
    print(f"The smallest eigenvalue is {smallest_eigenvalue:.2f}")



file_path = 'filtered_data.csv'
data = pd.read_csv(file_path)


data = data.select_dtypes(include=[np.number])
data = clean_data(data)


eigenvalues, anomalous_indices, normal_indices, lambda_min, lambda_max = apply_rmt(data.values)


summarize_results(eigenvalues, anomalous_indices, normal_indices, lambda_min, lambda_max)



data['Anomaly'] = 0
valid_anomalous_indices = [i for i in anomalous_indices if i < len(data)]
valid_anomalous_indices = [i for i in valid_anomalous_indices if i in data.index]  # To Ensure indices are in DataFrame's index
data.loc[valid_anomalous_indices, 'Anomaly'] = 1


X = data.drop('Anomaly', axis=1)
y = data['Anomaly']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


smote = SMOTE(sampling_strategy='minority',k_neighbors=3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Anomaly'])], axis=1)

balanced_data_path = 'balanced_data_with_rmt_smote.csv'
balanced_data = pd.read_csv(balanced_data_path)


balanced_data_multiplied = balanced_data * 100


balanced_data_multiplied.to_csv('balanced_data_with_rmt_smote_multiplied.csv', index=False)

