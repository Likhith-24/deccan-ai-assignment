# Import necessary libraries
from fastapi import FastAPI, HTTPException  # FastAPI framework and exception handling
from pydantic import BaseModel              # For input validation with Pydantic
import logging                              # For logging events
from model import predict                   # Imports prediction function
from utils import validate_input            # Imports validation function
import numpy as np                          # For array operations
import os                                   # For directory handling
import json                                 # For logging input data as JSON

# Ensure logs directory exists
log_dir = 'logs'                            # Defines the logs directory name
if not os.path.exists(log_dir):             # Checks if logs directory exists
    os.makedirs(log_dir)                    # Creates logs directory if it doesn’t exist

# Configure logging with rotation
logging.basicConfig(
    filename=os.path.join(log_dir, 'app.log'),  # Log file path
    level=logging.INFO,                     # Sets logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format with timestamp
    filemode='a'                            # Appends to file (default)
)
logger = logging.getLogger(__name__)        # Creates logger instance

# Initialize FastAPI app
app = FastAPI()                             # Creates FastAPI application instance

# Define input schema with Pydantic
class HouseFeatures(BaseModel):             # Defines expected input structure
    MedInc: float                           # Median income
    HouseAge: float                         # House age
    AveRooms: float                         # Average rooms
    AveBedrms: float                        # Average bedrooms
    Population: float                       # Population
    AveOccup: float                         # Average occupancy
    Latitude: float                         # Latitude
    Longitude: float                        # Longitude

# Startup event to log API start
@app.on_event("startup")                    # Runs when API starts
async def startup_event():
    logger.info("API startup initiated")    # Logs startup

# Define root endpoint
@app.get("/")                               # GET endpoint for root URL
async def root():
    logger.info("Root endpoint accessed")   # Logs access to root
    return {"message": "Welcome to the House Price Prediction API"}  # Returns welcome message

# Define prediction endpoint
@app.post("/predict")                       # POST endpoint for predictions
async def predict_price(house: HouseFeatures):
    """
    Predicts house price based on input features.
    Args:
        house (HouseFeatures): Input features in JSON format
    Returns:
        dict: Predicted price or error message
    """
    logger.info("Prediction request received")  # Logs request start
    try:
        features_dict = house.dict()            # Converts Pydantic model to dictionary
        logger.info(f"Input features: {json.dumps(features_dict)}")  # Logs input data
        features_array = validate_input(features_dict)  # Validates and converts to array
        prediction = predict(features_array)    # Makes prediction using model.py
        logger.info(f"Prediction made: {prediction}")  # Logs successful prediction
        return {"prediction": prediction}       # Returns prediction as JSON
    except ValueError as ve:                    # Catches validation errors
        logger.error(f"Validation error: {str(ve)}")  # Logs detailed error
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")  # Returns 400 error
    except FileNotFoundError as fnf:            # Catches model/scaler file errors
        logger.error(f"Model file error: {str(fnf)}")  # Logs file not found error
        raise HTTPException(status_code=500, detail="Model files missing")  # Returns 500 error
    except Exception as e:                      # Catches unexpected errors
        logger.error(f"Unexpected error: {str(e)}")  # Logs error with details
        raise HTTPException(status_code=500, detail="Internal server error")  # Returns 500 error