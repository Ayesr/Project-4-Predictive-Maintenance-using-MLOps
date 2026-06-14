 # PART 1: BUILD FASTAPI SERVICE
## Task 1.1: Create API Structure
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = "Predictive-Maintenance"
MODEL_STAGE = "Production"

model = None

# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info(
            f"Successfully loaded model: {MODEL_NAME} "
            f"({MODEL_STAGE})"
        )

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        model = None

    yield

    logger.info("Application shutdown completed")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine Learning API for Equipment Failure Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================
# Request Schema
# ============================================================

class EquipmentData(BaseModel):
    temperature: float = Field(..., gt=0, description="Temperature (°C)")
    vibration: float = Field(..., ge=0, description="Vibration Level")
    pressure: float = Field(..., gt=0, description="Pressure (bar)")

# ============================================================
# Health Check Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API Running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }

# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(data: EquipmentData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:
        start_time = datetime.now()

        input_df = pd.DataFrame([{
            "temperature": data.temperature,
            "vibration": data.vibration,
            "pressure": data.pressure
        }])

        prediction = model.predict(input_df)

        return {
            "prediction": int(prediction[0]),
            "timestamp": start_time.isoformat(),
            "model": MODEL_NAME
        }

    except Exception as e:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# ============================================================
# Example Run
# ============================================================

# uvicorn main:app --reload

## Task 1.2: Define Request/Response Models

from pydantic import BaseModel, Field
from typing import Annotated


# ============================================================
# Prediction Request Model
# ============================================================

class PredictionRequest(BaseModel):
    """
    Input features for equipment failure prediction.
    """

    temperature: Annotated[
        float,
        Field(
            ...,
            ge=-50,
            le=300,
            description="Equipment temperature in degrees Celsius (°C)"
        )
    ]

    vibration: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=100,
            description="Vibration level in mm/s"
        )
    ]

    pressure: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=500,
            description="Operating pressure in PSI"
        )
    ]

    rpm: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=50000,
            description="Rotational speed in revolutions per minute"
        )
    ]

    age_days: Annotated[
        int,
        Field(
            ...,
            ge=0,
            description="Number of days since last maintenance"
        )
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 85.0,
                "vibration": 0.7,
                "pressure": 110.0,
                "rpm": 1500.0,
                "age_days": 200
            }
        }
    }


# ============================================================
# Prediction Response Model
# ============================================================

class PredictionResponse(BaseModel):
    """
    API response returned after prediction.
    """

    will_fail: bool = Field(
        ...,
        description="Predicted equipment failure status"
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of equipment failure"
    )

    recommendation: str = Field(
        ...,
        description="Recommended maintenance action"
    )

    latency_ms: float = Field(
        ...,
        ge=0,
        description="Prediction response time in milliseconds"
    )

    timestamp: str = Field(
        ...,
        description="UTC timestamp when prediction was generated"
    )

    model_version: str = Field(
        default="1.0.0",
        description="Model version used for inference"
    )

    risk_level: str = Field(
        ...,
        description="Failure risk category: LOW, MEDIUM, HIGH, CRITICAL"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "will_fail": True,
                "probability": 0.91,
                "recommendation": "Immediate maintenance required",
                "latency_ms": 12.8,
                "timestamp": "2026-06-12T10:15:30Z",
                "model_version": "1.0.0",
                "risk_level": "CRITICAL"
            }
        }
    }

## Task 1.3: Create Endpoints

@app.get("/")
def root():
    return {
        "service": "Predictive Maintenance API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Equipment Failure"
)
def predict(request: PredictionRequest):

    if model is None:
        logger.error("Prediction requested but model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable"
        )

    start_time = time.perf_counter()

    try:
        # =====================================================
        # Create Feature DataFrame
        # =====================================================

        input_df = pd.DataFrame([{
            "temperature": request.temperature,
            "vibration": request.vibration,
            "pressure": request.pressure,
            "rpm": request.rpm,
            "age_days": request.age_days
        }])

        logger.info(
            "Prediction request received | "
            f"Temp={request.temperature}, "
            f"Vib={request.vibration}, "
            f"Pressure={request.pressure}, "
            f"RPM={request.rpm}, "
            f"Age={request.age_days}"
        )

        # =====================================================
        # Prediction
        # =====================================================

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(input_df)[0][1]
            )
        else:
            probability = float(
                model.predict(input_df)[0]
            )

        will_fail = probability >= 0.50

        # =====================================================
        # Risk Assessment
        # =====================================================

        if probability < 0.25:
            risk_level = "LOW"
            recommendation = "Continue normal operation"

        elif probability < 0.50:
            risk_level = "MEDIUM"
            recommendation = "Monitor equipment closely"

        elif probability < 0.75:
            risk_level = "HIGH"
            recommendation = "Schedule maintenance soon"

        else:
            risk_level = "CRITICAL"
            recommendation = "Immediate maintenance required"

        # =====================================================
        # Latency Calculation
        # =====================================================

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Prediction completed | "
            f"Failure={will_fail} | "
            f"Probability={probability:.4f} | "
            f"Risk={risk_level} | "
            f"Latency={latency_ms:.2f} ms"
        )

        return PredictionResponse(
            will_fail=will_fail,
            probability=round(probability, 4),
            recommendation=recommendation,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow().isoformat(),
            model_version="1.0.0",
            risk_level=risk_level
        )

    except Exception as e:

        logger.exception(
            f"Prediction failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )
# PART 2: ADD MONITORING
# Task 2.1: Add Metrics Tracking

from threading import Lock
from collections import deque
from typing import Dict, Any

class MetricsTracker:
    def __init__(self, max_samples: int = 1000):
        self._lock = Lock()
        self.total_requests = 0
        self.failures_predicted = 0
        self.errors = 0
        self.latencies = deque(maxlen=max_samples)
        self.predictions = deque(maxlen=max_samples) # Track recent predictions

    def update(self, latency_ms: float, probability: float, will_fail: bool, is_error: bool = False):
        with self._lock:
            self.total_requests += 1
            self.latencies.append(latency_ms)
            self.predictions.append(probability)
            
            if will_fail:
                self.failures_predicted += 1
            if is_error:
                self.errors += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            count = len(self.latencies)
            avg_latency = sum(self.latencies) / count if count > 0 else 0
            failure_rate = self.failures_predicted / self.total_requests if self.total_requests > 0 else 0
            
            return {
                'total_requests': self.total_requests,
                'failures_predicted': self.failures_predicted,
                'failure_rate': round(failure_rate, 3),
                'avg_latency_ms': round(avg_latency, 2),
                'errors': self.errors,
                'recent_predictions_count': len(self.predictions)
            }

# Initialize global tracker
tracker = MetricsTracker()

def predict(data):
    # ... your prediction logic ...
    # latency_ms = ...
    # probability = ...
    # will_fail = ...
    
    # Update metrics
    tracker.update(
        latency_ms=latency_ms, 
        probability=probability, 
        will_fail=will_fail
    )
    
    return {"result": "..."}


