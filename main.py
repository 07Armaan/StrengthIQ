from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import joblib
import pandas as pd

app = FastAPI(title="StrengthIQ Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load("final_pipeline.pkl")

class PredictRequest(BaseModel):
    Age: int
    Gender: str
    Supplement: str
    Weeks: int
    Initial_WT: float
    Primary_Benefit: str
    difference_found_in_WT: float

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/predict")
def predict(data: PredictRequest):
    input_df = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender,
        "Supplement": data.Supplement,
        "Weeks": data.Weeks,
        "Initial_WT": data.Initial_WT,
        "Primary_Benefit": data.Primary_Benefit,
        "difference_found_in_WT": data.difference_found_in_WT
    }])

    prediction = pipeline.predict(input_df)

    return {
        "Strength_Gain_pct": round(float(prediction[0]), 2)
    }