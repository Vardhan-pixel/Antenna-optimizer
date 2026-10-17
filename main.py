"""
Antenna Optimizer - FastAPI Backend
"""

import io
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from generate_antenna_data import generate_s11_curve
from report_generator import generate_report

MODEL_PATH = "best_antenna_model.pkl"
SCALER_PATH = "input_scaler.pkl"
METADATA_PATH = "model_metadata.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
metadata = joblib.load(METADATA_PATH)

INPUT_COLS = metadata["input_cols"]
OUTPUT_COLS = metadata["output_cols"]
MODEL_NAME = metadata["model_name"]

VALID_RANGES = {
    "frequency_GHz": (1.0, 10.0),
    "dielectric_constant": (2.0, 10.5),
    "substrate_height_mm": (0.5, 3.2),
}

app = FastAPI(
    title="Antenna Optimizer API",
    description="Predicts optimized microstrip patch antenna dimensions and performance using ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AntennaInput(BaseModel):
    frequency_GHz: float = Field(..., gt=0)
    dielectric_constant: float = Field(..., gt=0)
    substrate_height_mm: float = Field(..., gt=0)


class S11Point(BaseModel):
    frequency_GHz: float
    S11_dB: float


class AntennaPrediction(BaseModel):
    patch_width_mm: float
    patch_length_mm: float
    gain_dBi: float
    bandwidth_percent: float
    S11_min_dB: float
    s11_curve: list[S11Point]
    model_used: str
    warnings: list[str] = []


def validate_inputs(data: AntennaInput) -> list[str]:
    warnings = []
    checks = {
        "frequency_GHz": data.frequency_GHz,
        "dielectric_constant": data.dielectric_constant,
        "substrate_height_mm": data.substrate_height_mm,
    }
    for field, value in checks.items():
        lo, hi = VALID_RANGES[field]
        if not (lo <= value <= hi):
            warnings.append(
                f"{field}={value} is outside the training range [{lo}, {hi}]. "
                f"Prediction may be less reliable (extrapolation)."
            )
    return warnings


def run_prediction(data: AntennaInput):
    X = np.array([[data.frequency_GHz, data.dielectric_constant, data.substrate_height_mm]])
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)[0]
    return dict(zip(OUTPUT_COLS, y_pred))


@app.get("/")
def root():
    return {"message": "Antenna Optimizer API is running", "model": MODEL_NAME, "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=AntennaPrediction)
def predict(data: AntennaInput):
    try:
        warnings = validate_inputs(data)
        preds = run_prediction(data)

        s11_freqs, s11_vals = generate_s11_curve(
            fr_ghz=data.frequency_GHz,
            bandwidth_percent=preds["bandwidth_percent"],
            s11_min_db=preds["S11_min_dB"],
        )
        s11_curve = [
            S11Point(frequency_GHz=round(float(f), 4), S11_dB=round(float(s), 3))
            for f, s in zip(s11_freqs, s11_vals)
        ]

        return AntennaPrediction(
            patch_width_mm=round(float(preds["patch_width_mm"]), 4),
            patch_length_mm=round(float(preds["patch_length_mm"]), 4),
            gain_dBi=round(float(preds["gain_dBi"]), 3),
            bandwidth_percent=round(float(preds["bandwidth_percent"]), 3),
            S11_min_dB=round(float(preds["S11_min_dB"]), 3),
            s11_curve=s11_curve,
            model_used=MODEL_NAME,
            warnings=warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-report")
def predict_report(data: AntennaInput):
    try:
        warnings = validate_inputs(data)
        preds = run_prediction(data)

        s11_freqs, s11_vals = generate_s11_curve(
            fr_ghz=data.frequency_GHz,
            bandwidth_percent=preds["bandwidth_percent"],
            s11_min_db=preds["S11_min_dB"],
        )
        s11_curve = [
            {"frequency_GHz": float(f), "S11_dB": float(s)}
            for f, s in zip(s11_freqs, s11_vals)
        ]

        pdf_bytes = generate_report(
            inputs=data.dict(),
            predictions={k: round(float(v), 4) for k, v in preds.items()},
            s11_curve=s11_curve,
            model_name=MODEL_NAME,
            warnings=warnings,
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=antenna_design_report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
