# Coding Problem Difficulty Predictor(https://coding-problem-predictor.onrender.com)

A FastAPI app with a browser interface for predicting coding problem difficulty using a trained logistic regression model.

## Files

- `app.py` - API and static page routes
- `index.html` - Interactive prediction UI
- `logistic_model.pkl` - Trained classifier
- `scaler.pkl` - Fitted feature scaler required by the classifier
- `requirements.txt` - Python dependencies

Keep both `.pkl` files beside `app.py`. The scaler must be the fitted scaler from the same training run as the classifier.

## Run on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.
