from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI app
app = FastAPI(
    title="Coding Problem Difficulty Predictor",
    description="Predicts coding problem difficulty using Machine Learning",
    version="1.0"
)


# Load model and scaler from the project directory.
BASE_DIR = Path(__file__).resolve().parent
model = None
scaler = None
artifact_error = None

try:
    model = joblib.load(BASE_DIR / "logistic_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
except FileNotFoundError as error:
    artifact_error = f"Required model artifact not found: {error.filename}"


# Input schema
class ProblemInput(BaseModel):

    problem_length: float
    num_constraints: int
    num_tags: int
    implementation_steps: int

    has_tree: int
    has_graph: int
    has_heap: int
    has_recursion: int
    has_binary_search: int
    has_greedy: int
    has_dp: int


# Home route
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ready" if model is not None and scaler is not None else "model_unavailable",
        "detail": artifact_error,
    }


# Prediction route
@app.post("/predict")
def predict_difficulty(data: ProblemInput):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=artifact_error or "Model artifacts are unavailable.",
        )

    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "problem_length": data.problem_length,
        "num_constraints": data.num_constraints,
        "num_tags": data.num_tags,
        "implementation_steps": data.implementation_steps,

        "has_tree": data.has_tree,
        "has_graph": data.has_graph,
        "has_heap": data.has_heap,
        "has_recursion": data.has_recursion,
        "has_binary_search": data.has_binary_search,
        "has_greedy": data.has_greedy,
        "has_dp": data.has_dp
    }])


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)[0]


    # Probability
    probabilities = model.predict_proba(input_scaled)[0]

    class_names = model.classes_

    probability = {
        str(class_name): round(float(prob), 4)
        for class_name, prob in zip(class_names, probabilities)
    }


    return {
        "prediction": prediction,
        "probability": probability
    }