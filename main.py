"""
Personality Prediction API
Decryptogen Technical Assessment
FastAPI · scikit-learn · joblib
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import numpy as np
import pandas as pd
import joblib
import json
import os

# ── Load model & metadata ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "personality_model.joblib")
META_PATH  = os.path.join(BASE_DIR, "model", "model_meta.json")

pipeline = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

FEATURES = meta["features"]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Personality Prediction API",
    description=(
        "Predicts whether an individual is an **Introvert** or **Extrovert** "
        "from seven behavioural features using a trained ML pipeline. "
        f"Model: {meta['model_name']} · "
        f"Accuracy: {meta['accuracy']*100:.2f}% · "
        f"ROC-AUC: {meta['roc_auc']:.4f}"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    Time_spent_Alone: float = Field(
        ..., ge=0, le=11,
        description="Hours per day spent alone (0–11)",
        example=7.5,
    )
    Stage_fear: Literal["Yes", "No"] = Field(
        ..., description="Does the person have stage fear?", example="Yes"
    )
    Social_event_attendance: float = Field(
        ..., ge=0, le=10,
        description="Social events attended per month (0–10)",
        example=2.0,
    )
    Going_outside: float = Field(
        ..., ge=0, le=7,
        description="Times going outside per week (0–7)",
        example=2.0,
    )
    Drained_after_socializing: Literal["Yes", "No"] = Field(
        ..., description="Feels drained after socializing?", example="Yes"
    )
    Friends_circle_size: float = Field(
        ..., ge=0, le=20,
        description="Number of close friends (0–20)",
        example=3.0,
    )
    Post_frequency: float = Field(
        ..., ge=0, le=10,
        description="Social media posts per week (0–10)",
        example=2.0,
    )


class PredictionResponse(BaseModel):
    personality: Literal["Introvert", "Extrovert"]
    confidence: float = Field(..., description="Probability of the predicted class")
    probabilities: dict = Field(..., description="Class probabilities")


class HealthResponse(BaseModel):
    status: str
    model: str
    accuracy: str
    roc_auc: str
    cv_accuracy: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "Personality Prediction API — Decryptogen Technical Assessment",
        "model": meta["model_name"],
        "accuracy": f"{meta['accuracy']*100:.2f}%",
        "roc_auc": meta["roc_auc"],
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy",
        model=meta["model_name"],
        accuracy=f"{meta['accuracy']*100:.2f}%",
        roc_auc=str(meta["roc_auc"]),
        cv_accuracy=(
            f"{meta['cv_acc_mean']*100:.2f}% ± {meta['cv_acc_std']*100:.2f}%"
        ),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([{
            "Time_spent_Alone":           request.Time_spent_Alone,
            "Stage_fear":                 request.Stage_fear,
            "Social_event_attendance":    request.Social_event_attendance,
            "Going_outside":              request.Going_outside,
            "Drained_after_socializing":  request.Drained_after_socializing,
            "Friends_circle_size":        request.Friends_circle_size,
            "Post_frequency":             request.Post_frequency,
        }])

        proba   = pipeline.predict_proba(input_data)[0]
        pred_idx = int(np.argmax(proba))

        # target_map: {"0": "Extrovert", "1": "Introvert"}
        label_map = meta["target_map"]
        predicted = label_map[str(pred_idx)]
        confidence = float(proba[pred_idx])

        return PredictionResponse(
            personality=predicted,
            confidence=round(confidence, 4),
            probabilities={
                "Extrovert": round(float(proba[0]), 4),
                "Introvert": round(float(proba[1]), 4),
            },
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
