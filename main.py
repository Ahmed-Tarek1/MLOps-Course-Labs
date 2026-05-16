"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

import time
import os
from dotenv import load_dotenv
load_dotenv()

from litestar import Litestar, get, post
from pydantic import BaseModel
from app.logger_setup import setup_logging
from app.model_utils import FEATURE_COLUMNS, predict_churn

logger, axiom_client = setup_logging()
DATASET = os.getenv("AXIOM_DATASET")

# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@get("/")
async def home() -> dict:
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Churn Prediction API"}


@get("/health")
async def health() -> dict:
    return {"status": "healthy"}

@post("/predict")
async def predict(data: ChurnRequest) -> dict:
    start_time = time.time()
    features = [getattr(data, col) for col in FEATURE_COLUMNS]
    prediction = predict_churn(features)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Send structured event to Axiom
    axiom_client.ingest_events(dataset=DATASET, events=[{
        # Server metrics
        "response_time_ms": duration_ms,
        "endpoint": "/predict",
        "status_code": 201,

        # Model metrics
        "predicted_class": "Churn" if prediction == 1 else "No Churn",
        "churn": prediction == 1,

        # Data metrics
        "credit_score": data.CreditScore,
        "age": data.Age,
        "balance": data.Balance,
        "geography": data.Geography,
        "gender": data.Gender,
        "num_of_products": data.NumOfProducts,
        "is_active_member": data.IsActiveMember,
    }])

    return {"churn_prediction": prediction}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Litestar(
    route_handlers=[home, health, predict],
)
