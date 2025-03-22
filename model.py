# Import necessary libraries
import joblib                  # For loading saved model and scaler
import numpy as np             # For array operations

# Load the saved model and scaler
model = joblib.load('models/best_rf_model.pkl')  # Loads the trained Random Forest model
scaler = joblib.load('models/scaler.pkl')        # Loads the scaler used during training

# Define prediction function
def predict(features):
    """
    Makes a prediction given a numpy array of features.
    Args:
        features (np.ndarray): Array of shape (1, 8) with house features
    Returns:
        float: Predicted house price
    """
    features_scaled = scaler.transform(features)  # Scales input features using saved scaler
    prediction = model.predict(features_scaled)  # Predicts using the loaded model
    return prediction[0]                         # Returns single prediction value