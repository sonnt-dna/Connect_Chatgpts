from fastapi import FastAPI
import joblib
import numpy
from pydantic import BaseModel
app = FastAPI()

loaded model and scaler files

model = joblib.load('forest_model.jplib')
scaler = joblib.load('scaler.joblib')

class PredictionRequest(BaseModel):
    Generate: float
    Speed: float
    steam_flow: float
    pressure_after: float
    temp_after: float
@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.array([[request.Generate, request.Speed, request.steam_flow, request.pressure_after, request.temp_after']])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)