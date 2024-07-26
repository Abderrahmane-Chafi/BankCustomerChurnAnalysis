import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Load the trained model
model = joblib.load('best_pipeline_XGBoost_over.pkl')

# Define the input data format
class ChurnData(BaseModel):
    Age: int
    Balance: float
    IsActiveMember: int
    NumOfProducts: int
    France: int
    Germany: int
    Spain: int
    Female: int

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.post("/predict")
async def predict_churn(data: ChurnData):
    try:
        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
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
