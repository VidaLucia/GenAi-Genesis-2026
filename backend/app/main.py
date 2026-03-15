from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.fraud import router as fraud_router

app = FastAPI(
    title="SCOPE Fraud Detection API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fraud_router)
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}