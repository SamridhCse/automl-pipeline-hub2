import io
import logging
import os
import uuid
import shutil
from functools import lru_cache

import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Path
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score

from app.engine import run_automl_job, run_retraining_job
from app.schemas import PredictRequest, PredictResponse, TrainResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("automl.api")

app = FastAPI(title="AutoML Pipeline Hub", version="1.0.0")

JOB_STATUS = {}
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@lru_cache(maxsize=10)
def _load_model(model_name: str, version: str = "latest"):
    logger.info(f"Loading model: {model_name} v{version}")
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.pyfunc.load_model(model_uri)

def evaluate_predictions(y_true, y_pred, is_regression):
    if is_regression:
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))
        }
    else:
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            "F1_Score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }

@app.get("/automl/status/{job_id}")
def get_job_status(job_id: str):
    return JOB_STATUS.get(job_id, {"status": "not_found", "message": "Job not found"})

@app.get("/automl/models")
def list_registered_models():
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        client = MlflowClient(tracking_uri=tracking_uri)
        models = client.search_registered_models(max_results=1000)
        return {"models": sorted([m.name for m in models], reverse=True)}
    except Exception as e:
        return {"models": []}

@app.post("/automl/train", response_model=TrainResponse)
async def trigger_automl(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_col: str = Form(...),
    experiment_name: str = Form("Auto_Project"),
    n_trials: int = Form(10)
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # --- TASK A: Save original data for future retraining ---
    save_path = os.path.join(DATA_DIR, f"{experiment_name}_original.csv")
    df.to_csv(save_path, index=False)
    # -------------------------------------------------------

    job_id = str(uuid.uuid4())
    JOB_STATUS[job_id] = {"status": "running", "message": "Training initial model..."}
    background_tasks.add_task(run_automl_job, df, target_col, experiment_name, n_trials, job_id, JOB_STATUS)

    return TrainResponse(
        message="Training started.", job_id=job_id, experiment_name=experiment_name,
        registered_model_target=f"{experiment_name}_champion", n_trials=n_trials
    )

# --- TASK B: Evaluate Model on New Data ---
@app.post("/monitor/evaluate")
async def evaluate_new_batch(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    target_col: str = Form(...)
):
    try:
        df_new = pd.read_csv(io.BytesIO(await file.read()))
        pipeline = _load_model(model_name)
        
        if target_col not in df_new.columns:
            raise HTTPException(400, "Target column not found in new data")
            
        X_new = df_new.drop(columns=[target_col])
        y_true = df_new[target_col]
        y_pred = pipeline.predict(X_new)
        
        is_reg = pd.api.types.is_numeric_dtype(y_true) and y_true.nunique() > 15
        metrics = evaluate_predictions(y_true, y_pred, is_reg)
        
        # Report data summary
        summary = {
            "rows": len(df_new),
            "missing_values": int(df_new.isnull().sum().sum()),
            "duplicates": int(df_new.duplicated().sum()),
            "target_dist": df_new[target_col].value_counts().to_dict() if not is_reg else "Continuous"
        }
        return {"metrics": metrics, "summary": summary, "is_regression": is_reg}
    except Exception as e:
        raise HTTPException(500, str(e))

# --- TASK C & D: Retrain & Quality Gate ---
@app.post("/monitor/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    experiment_name: str = Form(...),
    target_col: str = Form(...),
    current_metric: str = Form(...), # e.g., 'F1_Score' or 'R2'
    n_trials: int = Form(5)
):
    original_data_path = os.path.join(DATA_DIR, f"{experiment_name}_original.csv")
    if not os.path.exists(original_data_path):
        raise HTTPException(400, "Original training data not found. Cannot retrain.")

    df_new = pd.read_csv(io.BytesIO(await file.read()))
    df_old = pd.read_csv(original_data_path)
    
    job_id = str(uuid.uuid4())
    JOB_STATUS[job_id] = {"status": "running", "message": "Retraining candidate model..."}
    
    # Background task to handle combine + train
    background_tasks.add_task(
        run_retraining_job, df_old, df_new, target_col, experiment_name, current_metric, n_trials, job_id, JOB_STATUS
    )
    return {"message": "Retraining started", "job_id": job_id}

@app.post("/automl/predict/{model_name}", response_model=PredictResponse)
async def predict(request: PredictRequest, model_name: str = Path(...)):
    try:
        pipeline = _load_model(model_name)
        input_df = pd.DataFrame(request.data)
        preds = pipeline.predict(input_df)
        preds_list = preds.tolist() if hasattr(preds, 'tolist') else preds
        return PredictResponse(model_name=model_name, predictions=preds_list)
    except Exception as e:
        raise HTTPException(500, str(e))