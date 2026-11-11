
import os
import json
import joblib
import pandas as pd

from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# --------------------------------------------------
# Load saved model artifacts
# --------------------------------------------------

MODEL_DIR = "artifacts"

model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
metadata_path = os.path.join(MODEL_DIR, "metadata.json")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

with open(metadata_path, "r") as file:
    metadata = json.load(file)

THRESHOLD = metadata["classification_threshold"]

# --------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="LightGBM-based transaction fraud detection service",
    version="1.0.0"
)

# --------------------------------------------------
# Define input schema using Pydantic
# --------------------------------------------------

class TransactionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    step: int = Field(gt=0)
    type: Literal[
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": metadata["model_name"],
        "version": app.version
    }


# --------------------------------------------------
# Fraud prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_fraud(transaction: TransactionRequest):

    try:

        # Convert validated input into DataFrame
        input_data = pd.DataFrame([{
            "step": transaction.step,
            "type": transaction.type,
            "amount": transaction.amount,
            "oldbalanceOrg": transaction.oldbalanceOrg,
            "oldbalanceDest": transaction.oldbalanceDest
        }])

        # Apply saved preprocessing
        processed_data = preprocessor.transform(input_data)

        # Generate fraud probability
        probability = float(
            model.predict_proba(processed_data)[0, 1]
        )

        # Apply validation-selected threshold
        prediction = int(probability >= THRESHOLD)

        return {
            "prediction": prediction,
            "classification": (
                "Fraudulent" if prediction == 1 else "Legitimate"
            ),
            "fraud_probability": round(probability, 6),
            "classification_threshold": round(THRESHOLD, 6),
            "model_name": metadata["model_name"]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
