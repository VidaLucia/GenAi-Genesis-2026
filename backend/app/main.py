from fastapi import FastAPI
from app.routes.fraud import router as fraud_router

app = FastAPI(
    title="Fraud Detection MVP",
    
)

app.include_router(fraud_router)


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}