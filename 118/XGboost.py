import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("IEEE118NormalWithPd_Qd.csv")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# XGBoost Regressor
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror')
xgb_regressor.fit(X_train_scaled, y_train)
# Save the trained XGBoost model
joblib.dump(xgb_regressor, 'xgb_regressor_model.pkl')
y_pred_xgb = xgb_regressor.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print("Mean Squared Error for XGBoost:", mse_xgb)


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

# Assuming y_test and y_pred_xgb are already defined and contain the true and predicted values respectively

# Initialize lists to store metrics for each output
accuracies = []
f1_scores = []
precisions = []
recalls = []
conf_matrices = []

# Calculate metrics for each output
for i in range(y_test.shape[1]):
	y_test_i = y_test.iloc[:, i].round()
	y_pred_i = y_pred_xgb[:, i].round()
	
	accuracies.append(accuracy_score(y_test_i, y_pred_i))
	f1_scores.append(f1_score(y_test_i, y_pred_i, average='weighted'))
	precisions.append(precision_score(y_test_i, y_pred_i, average='weighted'))
	recalls.append(recall_score(y_test_i, y_pred_i, average='weighted'))
	conf_matrices.append(confusion_matrix(y_test_i, y_pred_i))

# Print the metrics
print("Accuracies:", accuracies)
print("F1 Scores:", f1_scores)
print("Precisions:", precisions)
print("Recalls:", recalls)
print("Confusion Matrices:\n", conf_matrices)