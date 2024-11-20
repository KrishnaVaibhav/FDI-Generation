import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor as mor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score

import joblib
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = pd.read_csv("IEEE118NormalWithPd_Qd.csv")

# Identify inf values and replace with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Select features and target variables
vgm_columns = [col for col in data.columns if 'VGM' in col]
pg_columns = [col for col in data.columns if 'PG' in col]
pl_columns = [col for col in data.columns if 'PL' in col]
ql_columns = [col for col in data.columns if 'QL' in col]

vlm_columns = [col for col in data.columns if 'VLM' in col]
vla_columns = [col for col in data.columns if 'VLA' in col]
vga_columns = [col for col in data.columns if 'VGA' in col]

X = data[vgm_columns + pg_columns + pl_columns + ql_columns]
y = data[vlm_columns + vla_columns + vga_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create the Random Forest Regressor with parallel processing
mor_model = mor(RandomForestRegressor(n_estimators=100, random_state=123, n_jobs=16))

# Fit the model
print("Training the model...")
mor_model.fit(X_train, y_train)

# Make predictions
y_pred = mor_model.predict(X_test)

# Calculate accuracy for each output and average the results
accuracies = []
for i in range(y_test.shape[1]):
    accuracies.append(accuracy_score(y_test.iloc[:, i].round(), y_pred[:, i].round()))

average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average Accuracy: {average_accuracy}")

# Calculate F1 score for each output and average the results
f1_scores = []
for i in range(y_test.shape[1]):
    f1_scores.append(f1_score(y_test.iloc[:, i].round(), y_pred[:, i].round(), average='weighted'))

average_f1_score = sum(f1_scores) / len(f1_scores)
print(f"Average F1 Score: {average_f1_score}")

# Calculate precision for each output and average the results
precision_scores = []
for i in range(y_test.shape[1]):
    precision_scores.append(precision_score(y_test.iloc[:, i].round(), y_pred[:, i].round(), average='weighted'))

average_precision = sum(precision_scores) / len(precision_scores)
print(f"Average Precision: {average_precision}")

# Calculate recall for each output and average the results
recall_scores = []
for i in range(y_test.shape[1]):
    recall_scores.append(recall_score(y_test.iloc[:, i].round(), y_pred[:, i].round(), average='weighted'))

average_recall = sum(recall_scores) / len(recall_scores)
print(f"Average Recall: {average_recall}")

# Calculate confusion matrix for each output and plot the first one as an example
conf_matrix = confusion_matrix(y_test.iloc[:, 0].round(), y_pred[:, 0].round())
print(f"Confusion Matrix for first output:\n{conf_matrix}")

# Plot confusion matrix
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate Root Mean Squared Error
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")

# Save the model
print("Saving the model...")
joblib.dump(mor_model, 'random_forest_model_new.pkl')
