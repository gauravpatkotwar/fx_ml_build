"""FastAPI REST API for model serving"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.ensemble import EnsembleModel
from src.data.features import FeatureEngineer
from src.data.storage import DataStorage
from src.data.dataset import MLDatasetBuilder
from src.serving.health import router as health_router

logger = logging.getLogger(__name__)

app = FastAPI(title="FX ML Trading API", version="0.1.0")
app.include_router(health_router)

model = None
feature_engineer = FeatureEngineer()
storage = DataStorage(Path("data/processed"))

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "H1"
    bars: int = 500

class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    timestamp: str
    prediction: int
    signal: str
    confidence: float
    probabilities: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = EnsembleModel.load("models/ensemble_latest")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/predict", response_model=PredictionResponse)
async def get_prediction(symbol: str, timeframe: str = "H1", bars: int = 500):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = storage.read(symbol, timeframe)
        if df.empty or len(df) < 100:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")

        df = feature_engineer.create_features(df)
        feature_cols = [c for c in df.columns if c not in ["timestamp", "instrument", "granularity"]]
        latest = df[feature_cols].iloc[[-1]].fillna(method="ffill").fillna(0)

        prediction, probabilities = model.predict_with_proba(latest)
        confidence = float(max(probabilities[0]))

        signal_map = {0: "STRONG_SELL", 1: "SELL", 2: "HOLD", 3: "BUY", 4: "STRONG_BUY"}

        return PredictionResponse(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow().isoformat(),
            prediction=int(prediction),
            signal=signal_map.get(int(prediction), "UNKNOWN"),
            confidence=confidence,
            probabilities={
                "STRONG_SELL": float(probabilities[0][0]),
                "SELL": float(probabilities[0][1]),
                "HOLD": float(probabilities[0][2]),
                "BUY": float(probabilities[0][3]),
                "STRONG_BUY": float(probabilities[0][4])
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
async def get_feature_importance(top_n: int = 20):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance = model.get_feature_importance(top_n)
    return {"features": [{"feature": f, "importance": float(imp)} for f, imp in importance]}

@app.get("/model-info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model_type": "ensemble", "base_models": ["xgboost", "lightgbm", "random_forest"]}