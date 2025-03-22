# Import necessary libraries
import pandas as pd              # For DataFrame operations
import numpy as np               # For array operations
from sklearn.datasets import fetch_california_housing  # To load the dataset
from sklearn.model_selection import train_test_split   # To split data
from sklearn.preprocessing import StandardScaler       # To scale features
from sklearn.ensemble import RandomForestRegressor     # Chosen model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
from sklearn.model_selection import GridSearchCV       # For hyperparameter tuning
import joblib                  # To save model and scaler
import seaborn as sns          # For visualization
import matplotlib.pyplot as plt  # For plotting
import mlflow                  # For experiment tracking
import mlflow.sklearn          # For scikit-learn model logging

# Load the dataset
housing = fetch_california_housing(as_frame=True)  # Fetches dataset as a DataFrame
df = housing.frame                                # Converts to pandas DataFrame

# Exploratory Data Analysis (EDA)
print("Dataset Info:")                            # Prints basic info about dataset
print(df.info())                                  # Shows data types and non-null counts
print("\nMissing Values:")                        # Checks for missing values
print(df.isnull().sum())                          # Should be all zeros
print("\nDescriptive Statistics:")                # Shows summary statistics
print(df.describe())                              # Mean, std, min, max, etc.

# Visualize correlations
corr_matrix = df.corr()                           # Computes correlation matrix
plt.figure(figsize=(10, 8))                       # Sets figure size for heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  # Creates heatmap with correlation values
plt.title("Feature Correlation Heatmap")          # Adds title to plot
plt.savefig("correlation_heatmap.png")            # Saves plot as PNG
plt.close()                                       # Closes plot to free memory

# Prepare features and target
X = df.drop('MedHouseVal', axis=1)                # Features (all columns except target)
y = df['MedHouseVal']                             # Target variable (house price)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

# Scale the features
scaler = StandardScaler()                         # Initializes scaler
X_train_scaled = scaler.fit_transform(X_train)    # Fits scaler to training data and transforms it
X_test_scaled = scaler.transform(X_test)          # Transforms test data using same scaler

# Train initial Random Forest model
rf_model = RandomForestRegressor(random_state=42)  # Initializes RF with fixed random state
rf_model.fit(X_train_scaled, y_train)             # Trains model on scaled training data

# Evaluate initial model
y_pred = rf_model.predict(X_test_scaled)          # Makes predictions on test data
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Computes RMSE
mae = mean_absolute_error(y_test, y_pred)         # Computes MAE
r2 = r2_score(y_test, y_pred)                     # Computes R² score
print("\nInitial Random Forest Performance:")     # Prints evaluation header
print(f"RMSE: {rmse:.4f}")                        # Prints RMSE with 4 decimal places
print(f"MAE: {mae:.4f}")                          # Prints MAE
print(f"R²: {r2:.4f}")                            # Prints R²

# Hyperparameter tuning with GridSearchCV
param_grid = {                                    # Defines parameter grid for tuning
    'n_estimators': [100, 200],                   # Number of trees
    'max_depth': [None, 10, 20],                  # Maximum depth of trees
    'min_samples_split': [2, 5]                   # Minimum samples to split a node
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),  # Initializes GridSearchCV
                          param_grid=param_grid,     # Uses defined parameters
                          cv=5,                      # 5-fold cross-validation
                          scoring='neg_mean_squared_error',  # Scores based on negative MSE
                          n_jobs=-1)                 # Uses all CPU cores
grid_search.fit(X_train_scaled, y_train)          # Runs grid search on training data
best_rf_model = grid_search.best_estimator_       # Gets best model from grid search
print("\nBest Parameters from GridSearchCV:")     # Prints best parameters header
print(grid_search.best_params_)                   # Prints best parameters found

# Evaluate optimized model
y_pred_best = best_rf_model.predict(X_test_scaled)  # Makes predictions with optimized model
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)  # Computes RMSE
mae_best = mean_absolute_error(y_test, y_pred_best)  # Computes MAE
r2_best = r2_score(y_test, y_pred_best)           # Computes R²
print("\nOptimized Random Forest Performance:")   # Prints evaluation header
print(f"RMSE: {rmse_best:.4f}")                   # Prints RMSE
print(f"MAE: {mae_best:.4f}")                     # Prints MAE
print(f"R²: {r2_best:.4f}")                       # Prints R²

# Log with MLflow
with mlflow.start_run():                          # Starts an MLflow run
    mlflow.log_param("n_estimators", grid_search.best_params_['n_estimators'])  # Logs n_estimators
    mlflow.log_param("max_depth", grid_search.best_params_['max_depth'])        # Logs max_depth
    mlflow.log_param("min_samples_split", grid_search.best_params_['min_samples_split'])  # Logs min_samples_split
    mlflow.log_metric("rmse", rmse_best)          # Logs RMSE metric
    mlflow.log_metric("mae", mae_best)            # Logs MAE metric
    mlflow.log_metric("r2", r2_best)              # Logs R² metric
    mlflow.sklearn.log_model(best_rf_model, "model")  # Logs the trained model
    mlflow.log_artifact("correlation_heatmap.png")  # Logs the correlation heatmap

# Save the model and scaler
joblib.dump(best_rf_model, 'models/best_rf_model.pkl')  # Saves optimized model to file
joblib.dump(scaler, 'models/scaler.pkl')          # Saves scaler to file
print("\nModel and scaler saved to 'models/' directory.")  # Confirms save
print("Model logged with MLflow.")                # Confirms MLflow logging