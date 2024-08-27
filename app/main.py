from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import os
import pandas as pd
import sklearn
import fastapi
import uvicorn
import sklearn
import mlflow
import pydantic

# Set the MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Abderrahmane-Chafi/BankCustomerChurnAnalysis.mlflow"

# Load the model from MLflow
run_id = "d550c2df913b41278a106225d91a1e3f"
model_uri = f"runs:/{run_id}/sklearn-model"

# Debug prints
print(f"FastAPI: {fastapi.__version__}")
print(f"Uvicorn: {uvicorn.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"MLflow: {mlflow.__version__}")
print(f"Pydantic: {pydantic.__version__}")

print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("Model URI:", model_uri)

try:
    model = mlflow.sklearn.load_model(model_uri=model_uri).best_estimator_
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None

app = FastAPI()

# Define the input data model
class ChurnData(BaseModel):
    Age: int
    Balance: float
    IsActiveMember: int
    NumOfProducts: int
    France: int
    Germany: int
    Spain: int
    Female: int



@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}


@app.post("/predict")
async def predict_churn(data: ChurnData):
    try:
        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([data.model_dump()]) #Why Use a DataFrame? Consistency with Model Input Requirements: Many machine learning models expect input data in a tabular format like a DataFrame.
        
        # Make prediction
        prediction = model.predict(input_df)
        if(int(prediction[0])==1):
            prediction="Retained"
        else:
            prediction="Exited"
        # Return the prediction
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

