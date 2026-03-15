from fastapi import APIRouter, HTTPException
from app.schemas.fraud_schema import (
    TransactionRequest,
    FraudResponse,
    ReviewLogRequest
)
from app.services.profile_service import load_user_profile
from app.services.feature_service import build_features
from app.services.model_service import predict_model_scores
from app.services.scoring_service import compute_final_decision
from app.services.log_service import save_analyst_log, list_analyst_logs, review_log_status, list_user_transactions

router = APIRouter(prefix="", tags=["fraud"])


@router.post("/score-transaction", response_model=FraudResponse)
def score_transaction(payload: TransactionRequest):
    profile = load_user_profile(payload.user_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="User profile not found")

    features = build_features(payload, profile)
    model_scores = predict_model_scores(features)
    decision = compute_final_decision(features, model_scores)

    log_id = save_analyst_log(
        transaction=payload.model_dump(),
        features=features,
        model_outputs=model_scores,
        decision=decision
    )

    return FraudResponse(
        fraud_probability=decision["final_score"],
        label=decision["label"],
        reasons=decision["reasons"],
        heatmap_score=decision["heatmap_score"],
        ml_score=decision["ml_score"],
        final_score=decision["final_score"],
        analyst_log_id=log_id
    )


@router.get("/user-profile/{user_id}")
def get_user_profile(user_id: str):
    profile = load_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="User profile not found")
    return profile


@router.get("/analyst-logs")
def get_logs():
    return {"logs": list_analyst_logs()}


@router.get("/transactions/{user_id}")
def get_user_transactions(user_id: str):
    return {"transactions": list_user_transactions(user_id)}


@router.post("/review-log/{log_id}")
def review_log(log_id: str, payload: ReviewLogRequest):
    updated = review_log_status(log_id, payload.review_status)
    if not updated:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"message": "Log updated", "log_id": log_id, "review_status": payload.review_status}