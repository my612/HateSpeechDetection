from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict
from pydantic import BaseModel
from typing import List
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: bool
    prob: float
    tokens: List[str]

app = FastAPI(
    title="Hate Speech Detection API",
    description="API for detecting hate speech in text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Hate Speech Detection API!",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """
    Endpoint to predict hate speech in a given text.
    """
    try:
        logger.info(f"Processing prediction request for text: {request.text}")
        result = predict(request.text)
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat()
    }
