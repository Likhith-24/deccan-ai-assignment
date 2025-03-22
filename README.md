# deccan-ai-assignment

A machine learning project to predict house prices using the California Housing Dataset, deployed as a REST API with FastAPI, MLflow versioning, and a Streamlit frontend.

## Setup
1. Clone the repository: `git clone <repo_url>`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Training the Model
- Run `python train.py` to train the Random Forest model, log it with MLflow, and save to `models/`.
- Outputs: `best_rf_model.pkl`, `scaler.pkl`, `mlruns/`, `correlation_heatmap.png`.

## Running the API
- Start the server: `uvicorn app:app --reload` or use Docker:
  ```bash
  docker build -t house-price-api .
  docker run -p 8000:8000 house-price-api
  Access at http://127.0.0.1:8000/ ```

## Running the Frontend
- Ensure the API is running.
- Run streamlit run frontend.py and visit http://localhost:8501.

## API Usage
- GET /: curl http://127.0.0.1:8000/
- POST /predict: curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}'
## Approach
- Preprocessing: Scaled features with StandardScaler.
- Model: Random Forest optimized with GridSearchCV, logged with MLflow.
- Deployment: FastAPI with enhanced logging (logs/app.log) and error handling, Dockerized.
- Frontend: Streamlit UI for user interaction.

### Testing
1. **Run API:** `docker run -p 8000:8000 house-price-api`
2. **Run Frontend:** `streamlit run frontend.py`
3. **Test:** Input values in the UI, click "Predict", and verify predictions match CURL results.
4. **Check Logs:** Open `logs/app.log` to see detailed entries.

---

### Conclusion
- **Docker:** Updated to include MLflow, ensuring portability.
- **Logging/Error Handling:** Enhanced in `app.py` with input logging and specific error cases.
- **Frontend:** Streamlit UI (`frontend.py`) provides a simple interface.