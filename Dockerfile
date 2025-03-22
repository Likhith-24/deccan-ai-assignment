# Use official Python 3.11 slim image as base
 # Lightweight Python image with Python 3.11
FROM python:3.11-slim         

# Set working directory inside container
# All subsequent commands run in /app
WORKDIR /app                   

# Copy requirements file to container
# Copies updated requirements.txt
COPY requirements.txt .        

# Install dependencies
# Installs all packages, including MLflow
RUN pip install -r requirements.txt  

# Copy entire project to container
# Copies all files (app.py, model.py, models/, etc.)
COPY . .                       

# Expose port for the API
# Documents that port 8000 will be used
EXPOSE 8000                    

# Command to run the API
# Runs Uvicorn on all interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  