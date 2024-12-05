import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor as RF

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



# Function to handle inf values (consider replacing with a more domain-specific approach if applicable)
def handle_inf_values(df):
    for col in df.columns:
        df.loc[df[col] == np.inf, col] = df[col].mean()  # Replace inf with mean (adjustable)
        df.loc[df[col] == -np.inf, col] = -df[col].mean()  # Replace -inf with negative of mean
    return df


# Handle inf values in both training and testing data
data = handle_inf_values(data.copy())

X = data[vgm_columns + pg_columns + pl_columns + ql_columns]
y = data[vlm_columns + vla_columns + vga_columns]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define LightGBM parameter grid
lgb_param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 8],
}

# Create and train LightGBM model with GridSearchCV
lgb_regressor =MultiOutputRegressor(RF(n_estimators=100, n_jobs=2, max_depth=5))
print("Training LightGBM Model...")
lgb_regressor.fit(X_train,y_train)

# # Create and train LightGBM model with GridSearchCV
# lgb_regressor = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, n_jobs=2, max_depth=5))
# print("Training LightGBM Model...")
# lgb_regressor.fit(X_train,y_train)

# # Create and train LightGBM model with GridSearchCV
# lgb_regressor =MultiOutputRegressor(XGBRegressor(n_estimators=100, n_jobs=2, max_depth=5))
# print("Training LightGBM Model...")
# lgb_regressor.fit(X_train,y_train)


#pickle.dump(lgb_regressor,"lgb_regressor.pkl")
y_pred_lgb = lgb_regressor.predict(X_test)

# Evaluate LightGBM Model
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

print("LightGBM Model Evaluation:")
print("Mean Squared Error:", mse_lgb)
print("Mean Absolute Error:", mae_lgb)
print("R-squared Score:", r2_lgb)

print("Saving LightGBM Model...")
with open('RFmodel.pickle', 'wb') as handle:
    pickle.dump(lgb_regressor, handle) 
