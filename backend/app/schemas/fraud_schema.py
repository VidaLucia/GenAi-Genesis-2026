from typing import List, Literal
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    user_id: str
    transaction_id: str
    amount: float = Field(..., gt=0)
    merchant_name: str
    merchant_category: str
    transaction_lat: float
    transaction_lng: float
    phone_lat: float
    phone_lng: float
    timestamp: str


class FraudResponse(BaseModel):
    fraud_probability: float
    label: str
    reasons: List[str]
    heatmap_score: float
    ml_score: float
    final_score: float
    analyst_log_id: str


class ReviewLogRequest(BaseModel):
    review_status: Literal["pending", "confirmed_fraud", "false_positive", "legitimate"]