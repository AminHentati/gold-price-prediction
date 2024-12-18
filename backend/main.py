from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import os
import uvicorn

# Authenticate MLflow with DagsHub credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "hentatiamin0"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a0fb37da71b9753d88c2059e20e9a1c656209fc8"

mlflow.set_tracking_uri("https://dagshub.com/hentatiamin0/gold-price-prediction.mlflow")
import os

print("Username:", os.getenv("MLFLOW_TRACKING_USERNAME"))
print("Token:", os.getenv("MLFLOW_TRACKING_PASSWORD"))

# Initialize FastAPI
app = FastAPI()

print("MLflow tracking URI:", mlflow.get_tracking_uri())

# Try to load the model from MLflow
try:
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    df_mlflow = mlflow.search_runs(
        experiment_ids=all_experiments, filter_string="metrics.RMSE < 10"
    )

    if not df_mlflow.empty:
        run_id = df_mlflow.loc[df_mlflow['metrics.RMSE'].idxmax()]['run_id']
        logged_model = f"runs:/{run_id}/models"
        model = mlflow.pyfunc.load_model(logged_model)
        print(f"Model successfully loaded from run ID: {run_id}")
    else:
        raise ValueError("No runs found with the specified filter.")

except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Define the data model for prediction input
class PredictionData(BaseModel):
    Open: float
    High: float
    Low: float
    Volume: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Gold Price Prediction API"}

# Endpoint for CSV file predictions
@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check logs for errors."}

    try:
        data = pd.read_csv(file.file)
        data = data[['Open', 'High', 'Low', 'Volume']].fillna(0)
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Endpoint for JSON data predictions
@app.post("/predict")
def predict_from_json(data: PredictionData):
    if model is None:
        return {"error": "Model not loaded. Please check logs for errors."}

    try:
        received_data = data.dict()
        input_data = pd.DataFrame([received_data])
        predictions = model.predict(input_data)
        return {"prediction": predictions[0]}
    except Exception as e:
        return {"error": str(e)}

# Debug endpoint to verify environment variables and model tracking connection
@app.get('/debug')
async def debug():
    return {
        "MLFLOW_URI": mlflow.get_tracking_uri(),
        "USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME"),
        "TOKEN": os.getenv("MLFLOW_TRACKING_PASSWORD")
    }

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)