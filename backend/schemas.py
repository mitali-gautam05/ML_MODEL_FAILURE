from pydantic import BaseModel
from typing import Optional, Dict, Any

class MetricsOut(BaseModel):
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   Optional[float] = None

class PredictRequest(BaseModel):
    model_name: str
    features:   list[float]

class PredictResponse(BaseModel):
    model:      str
    prediction: int
    label:      str
    confidence: Optional[float] = None