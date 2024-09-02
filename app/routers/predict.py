from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.prediction_service import predict_call_result

router = APIRouter()

class PredictionRequest(BaseModel):
    input_texts: str

class PredictionResponse(BaseModel):
    category: str

@router.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    try:
        category= predict_call_result(request.input_texts)
        return PredictionResponse(category=category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
