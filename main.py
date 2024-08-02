from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import os
import pandas as pd
import sklearn

# Set the MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Abderrahmane-Chafi/BankCustomerChurnAnalysis.mlflow"

# Load the model from MLflow
run_id = "c853c1833e6748f2bea280b1f8767c30"
model_uri = f"runs:/{run_id}/sklearn-model"

# Debug prints
print(sklearn.__version__)
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
    #Uvicorn is a web server. It handles network communication - receiving requests from client applications 
    # such as users' browsers and sending responses to them. It communicates with FastAPI using the Asynchronous 
    # Server Gateway Interface (ASGI) - ASGI: a calling convention for web servers to forward requests to asynchronous-capable Python frameworks, and applications.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

