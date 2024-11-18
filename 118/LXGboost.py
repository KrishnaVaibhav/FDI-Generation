import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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

# Assuming you want to predict the first column of y
y = y.iloc[:, 0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to handle inf values (consider replacing with a more domain-specific approach if applicable)
def handle_inf_values(df):
    for col in df.columns:
        df.loc[df[col] == np.inf, col] = df[col].mean()  # Replace inf with mean (adjustable)
        df.loc[df[col] == -np.inf, col] = -df[col].mean()  # Replace -inf with negative of mean
    return df

# Handle inf values in both training and testing data
X_train = handle_inf_values(X_train.copy())
X_test = handle_inf_values(X_test.copy())

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define LightGBM parameter grid
lgb_param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 8],
}

# Create and train LightGBM model with GridSearchCV
lgb_regressor = lgb.LGBMRegressor(objective='regression', metric='mse')
lgb_grid_search = GridSearchCV(lgb_regressor, lgb_param_grid, cv=5, scoring='neg_mean_squared_error')
lgb_grid_search.fit(X_train_scaled, y_train)

# Get the best LightGBM model and predictions
best_lgb_model = lgb_grid_search.best_estimator_
y_pred_lgb = best_lgb_model.predict(X_test_scaled)

# Save the best LightGBM model in both .txt and .pkl formats
best_lgb_model.booster_.save_model('best_lgb_model.txt')
with open('best_lgb_model.pkl', 'wb') as f:
    pickle.dump(best_lgb_model, f)

# Evaluate LightGBM Model
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print("LightGBM Model Evaluation:")
print("Mean Squared Error:", mse_lgb)
print("Mean Absolute Error:", mae_lgb)
print("R-squared Score:", r2_lgb)


threshold = 0.5  # Adjust threshold based on your data and task
y_pred_lgb_class = (y_pred_lgb > threshold).astype(int)

# Binarize y_test based on the same threshold
y_test_class = (y_test > threshold).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_test_class, y_pred_lgb_class)
f1 = f1_score(y_test_class, y_pred_lgb_class, average='weighted')  # Weighted F1 score for imbalanced classes (consider other options)
precision = precision_score(y_test_class, y_pred_lgb_class, average='weighted')  # Weighted precision
recall = recall_score(y_test_class, y_pred_lgb_class, average='weighted')  # Weighted recall


# Print classification metrics
print("\nClassification Evaluation:")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Optionally, calculate confusion matrix
cm = confusion_matrix(y_test_class, y_pred_lgb_class)
print("\nConfusion Matrix:")
print(cm)