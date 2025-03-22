# 🏡 Deccan AI - House Price Prediction

A machine learning project to predict house prices using the California Housing Dataset, deployed as a REST API with FastAPI, MLflow versioning, and a Streamlit frontend.

## 📌 Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Running the API](#running-the-api)
- [Running the Frontend](#running-the-frontend)
- [API Usage](#api-usage)
- [Approach](#approach)
- [Testing](#testing)
- [Conclusion](#conclusion)

## 🔍 Overview
This project leverages **machine learning** to predict house prices using the **California Housing Dataset**. The model is trained using **Random Forest**, logged with **MLflow**, deployed using **FastAPI**, and comes with a **Streamlit** frontend.

## ⚙️ Setup
1. Clone the repository:  
   ```bash
   git clone <repo_url>
   ```
2. Create a virtual environment:  
   ```bash
   python -m venv venv
   ```
3. Activate it:  
   - Unix/macOS:  
     ```bash
     source venv/bin/activate
     ```
   - Windows:  
     ```bash
     venv\Scripts\activate
     ```
4. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Training the Model
- Run the training script:
  ```bash
  python train.py
  ```
- Outputs:
  - `best_rf_model.pkl`
  - `scaler.pkl`
  - `mlruns/`
  - `correlation_heatmap.png`

## 🚀 Running the API
Start the API server:
```bash
uvicorn app:app --reload
```
Or using Docker:
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```
API accessible at: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## 🎨 Running the Frontend
- Ensure the API is running.
- Launch Streamlit UI:
  ```bash
  streamlit run frontend.py
  ```
- Access at: [http://localhost:8501](http://localhost:8501)

## 🌐 API Usage
- **GET /** :
  ```bash
  curl http://127.0.0.1:8000/
  ```
- **POST /predict** :
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}'
  ```

## 🏗️ Approach
- **Preprocessing:** StandardScaler for feature scaling.
- **Model:** Optimized **Random Forest** with **GridSearchCV**, logged with MLflow.
- **Deployment:** FastAPI with logging (`logs/app.log`) and error handling, Dockerized.
- **Frontend:** Streamlit UI for user interaction.

## 🧪 Testing
1. **Run API:**  
   ```bash
   docker run -p 8000:8000 house-price-api
   ```
2. **Run Frontend:**  
   ```bash
   streamlit run frontend.py
   ```
3. **Test Predictions:** Input values in the UI, click "Predict", and verify results match **CURL** responses.
4. **Check Logs:**  
   ```bash
   cat logs/app.log
   ```

## 🎯 Conclusion
- ✅ **Dockerized MLflow integration** ensures model versioning & reproducibility.
- ✅ **Enhanced logging & error handling** in `app.py`.
- ✅ **Interactive Streamlit frontend** for seamless user experience.

📌 **Contributors:**  
- [@Likhith](https://github.com/Likhith-24)  

🌟 **Give a ⭐ if you like this project!**
