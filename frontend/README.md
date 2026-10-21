# Streamlit frontend

This directory contains the Streamlit business interface for the Telco Customer Churn API.

## Run locally

From the project root, activate the requested environment:

```powershell
conda activate ai
cd C:\Amulya\Projects\Telco-Customer-Churn
python -m pip install -r frontend\requirements-frontend.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
conda activate ai
cd C:\Amulya\Projects\Telco-Customer-Churn
streamlit run frontend\streamlit_app.py
```

The frontend defaults to `http://127.0.0.1:8000`. For a separately deployed backend:

```powershell
$env:CHURN_API_URL = "https://your-api.example.com"
streamlit run frontend\streamlit_app.py
```

## Backend contract

The frontend calls:

- `GET /health` to show connectivity, selected model, threshold, and current saved-model metrics.
- `POST /predict` to score the form's raw customer fields.

The API is the source of truth for model predictions and evaluation values shown in the Model information page. The frontend does not load the model artifact directly.

## UI behavior

- Shows an explicit backend connection state in the sidebar.
- Displays the active model and decision threshold from `/health`.
- Sends form data to `/predict` with a timeout.
- Handles validation errors, unavailable backends, timeouts, and other request failures visibly.
- Displays probability, risk classification, and the active decision threshold.

