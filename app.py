from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from mangum import Mangum

# Define the request body using Pydantic
class PredictionRequest(BaseModel):
    age: int
    family_history_with_overweight: int
    FAVC: int
    CAEC: int
    SCC: int

# Initialize FastAPI app
app = FastAPI()
handler = Mangum(app)

# Load the scaler and neural network model from the pickle file
with open('models/rf_model.pkl', 'rb') as file:
    model, threshold = pickle.load(file)

# Define the route to make predictions
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert request data to numpy array
        input_data = np.array([
            request.age,
            request.family_history_with_overweight,
            request.FAVC,
            request.CAEC,
            request.SCC
        ]).reshape(1, -1)

        # Get the prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        # Determine the final prediction based on the threshold
        final_prediction = (probabilities[1] >= threshold).astype(int)

        # Return the prediction and probabilities
        return {
            "prediction": final_prediction.tolist(),
            "probabilities": probabilities.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)