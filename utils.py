# Import necessary libraries
import numpy as np             # For array operations

# Define input validation function
def validate_input(features_dict):
    """
    Validates input features dictionary and converts to numpy array.
    Args:
        features_dict (dict): Dictionary with 8 house features
    Returns:
        np.ndarray: Validated and reshaped feature array
    Raises:
        ValueError: If input is invalid
    """
    expected_keys = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'Latitude', 'Longitude']  # Expected feature names
    if not all(key in features_dict for key in expected_keys):  # Checks if all keys are present
        missing = [k for k in expected_keys if k not in features_dict]  # Lists missing keys
        raise ValueError(f"Missing features: {missing}")           # Raises error if keys missing
    
    features = [features_dict[key] for key in expected_keys]       # Extracts values in order
    features_array = np.array(features).reshape(1, -1)             # Converts to 1x8 array
    
    if not np.isfinite(features_array).all():                      # Checks for NaN or infinite values
        raise ValueError("Input contains invalid values (NaN or infinite)")  # Raises error
    
    return features_array                                          # Returns validated array