from fastapi import FastAPI, HTMPException
from pydantic import BaseModel
import joblib
import numpy

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

// Initialize the FastAPI app app app
app = FastAPI()

# Define the request body
class PredictRequest(BaseModel):
    DEPTH: float
    CALI: float
    DT: float
    GR: float
    LLD: float
    LLS: float
    NPHI: float
    PHIE: float
    RHOB: float
    SW: float

# Define the prediction endpoint
@app.post('/predict')
def predict(request: PredictRequest):
    try:
        // Extract data from request
        data = np.array([request.DEPTN,request.CALI,request.DT,request.GR, request.LLD,request.LLS,request.NPHI, request.PHIE,request.RHOB,request.SW])
        // Make prediction
        prediction = model.predict(data)
        // Return the prediction
        return {'VWCL': prediction[0]
}
    except Exception as e:
        throw HTMPException(status_code=400, detail=str(e))


// Run the app
if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
