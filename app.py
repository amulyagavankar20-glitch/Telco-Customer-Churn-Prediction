from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PIPELINE = joblib.load(ARTIFACTS / "telco_churn_pipeline.joblib")
METADATA = json.loads(
    (ARTIFACTS / "telco_churn_metadata.json").read_text(encoding="utf-8")
)
THRESHOLD = float(METADATA["decision_threshold"])

SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

app = FastAPI(title="Telco Customer Churn API", version="1.0.0")


class CustomerPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str | float


def prepare_customer_records(records: pd.DataFrame) -> pd.DataFrame:
    records = records.drop(columns=["customerID", "Churn"], errors="ignore").copy()
    missing = set(REQUIRED_COLUMNS).difference(records.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    records["tenure"] = pd.to_numeric(records["tenure"], errors="coerce")
    records["MonthlyCharges"] = pd.to_numeric(records["MonthlyCharges"], errors="coerce")
    records["TotalCharges"] = pd.to_numeric(
        records["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    records.loc[
        records["tenure"].eq(0) & records["TotalCharges"].isna(), "TotalCharges"
    ] = 0.0

    records["TotalServices"] = records[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    records["AvgMonthlySpend"] = np.where(
        records["tenure"].gt(0),
        records["TotalCharges"] / records["tenure"].replace(0, np.nan),
        records["MonthlyCharges"],
    )
    records["AvgMonthlySpend"] = records["AvgMonthlySpend"].replace(
        [np.inf, -np.inf], np.nan
    )

    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    if records[numeric].isna().any().any():
        raise ValueError("Numeric fields contain invalid or missing values.")
    return records


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Telco Customer Churn API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": METADATA["selected_model"],
        "decision_threshold": THRESHOLD,
        "test_metrics": METADATA.get("test_metrics_tuned_threshold", {}),
        "versions": METADATA.get("versions", {}),
    }


@app.post("/predict")
def predict(payload: CustomerPayload) -> dict[str, Any]:
    try:
        prepared = prepare_customer_records(pd.DataFrame([payload.model_dump()]))
        probability = float(PIPELINE.predict_proba(prepared)[0, 1])
        return {
            "churn_probability": round(probability, 6),
            "predicted_churn": int(probability >= THRESHOLD),
            "decision_threshold": THRESHOLD,
            "risk_level": "High" if probability >= THRESHOLD else "Low",
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
