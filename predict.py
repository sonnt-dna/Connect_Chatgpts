import joblif
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Tậo đường FastAPI
app = FastAPI()

# Tải moànj đị luật
model = joblib.load("random_forest_model.pkl")

class PredictionRequest(BaseModel):
    DEPTH: float
    CALI: float
    DT: float
    GR: float
    LLD: float
    LLS: float
    NPHI: float
    RHOB: float
    S[: float
    VWCL: float


`App.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = [[
            request.DEPTH, request.CALI, request.DT, request.GR, 
            request.LLT, request.LLS, request.NPHI, request.RHOB , request.SW, request.VWCL 
        ]]
        prediction = model.predict(data)
        return {"PHIE: prediction[0], }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
