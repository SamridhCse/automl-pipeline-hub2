
import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mlflow.set_tracking_uri(MLFLOW_URI)


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    num_cols = X.select_dtypes(
        include=["int64", "float64", "number"]
    ).columns.tolist()

    cat_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    transformers = []

    if num_cols:
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers)


# ============================================================
# MODEL INSTANTIATION
# ============================================================

def instantiate_model(
    classifier_name: str,
    params: dict,
    is_regression: bool
):

    if is_regression:

        if classifier_name == "LinearReg":
            return LinearRegression()

        elif classifier_name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=params.get("rf_n_estimators", 100),
                max_depth=params.get("rf_max_depth", 10),
                random_state=42
            )

        elif classifier_name == "SVR":
            return SVR(
                C=params.get("svm_c", 1.0)
            )

        elif classifier_name == "KNN":
            return KNeighborsRegressor(
                n_neighbors=params.get("knn_n_neighbors", 5)
            )

        elif classifier_name == "LightGBM":
            return LGBMRegressor(
                learning_rate=params.get("lgbm_lr", 0.1),
                num_leaves=params.get("lgbm_num_leaves", 31),
                random_state=42,
                verbose=-1
            )

        else:
            return XGBRegressor(
                learning_rate=params.get("xgb_lr", 0.1),
                max_depth=params.get("xgb_max_depth", 6),
                random_state=42
            )

    else:

        if classifier_name == "LogReg":
            return LogisticRegression(
                C=params.get("logreg_c", 1.0),
                max_iter=500,
                random_state=42
            )

        elif classifier_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=params.get("rf_n_estimators", 100),
                max_depth=params.get("rf_max_depth", 10),
                random_state=42
            )

        elif classifier_name == "SVM":
            return SVC(
                C=params.get("svm_c", 1.0),
                probability=True,
                random_state=42
            )

        elif classifier_name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=params.get("knn_n_neighbors", 5)
            )

        elif classifier_name == "LightGBM":
            return LGBMClassifier(
                learning_rate=params.get("lgbm_lr", 0.1),
                num_leaves=params.get("lgbm_num_leaves", 31),
                random_state=42,
                verbose=-1
            )

        else:
            return XGBClassifier(
                learning_rate=params.get("xgb_lr", 0.1),
                max_depth=params.get("xgb_max_depth", 6),
                eval_metric="logloss",
                random_state=42
            )


# ============================================================
# AUTOML TRAINING JOB
# ============================================================

def run_automl_job(
    df: pd.DataFrame,
    target_col: str,
    experiment_name: str,
    n_trials: int = 10,
    job_id=None,
    job_status_dict=None,
    override_name=None
):

    X = df.drop(columns=[target_col])

    # Remove useless ID columns
    cols_to_drop = [
        col for col in X.columns
        if col.lower() == "id"
        or col.lower().endswith("_id")
    ]

    if cols_to_drop:

        print(
            f"\n[DATA CLEANING] "
            f"Dropping ID columns: {cols_to_drop}"
        )

        X = X.drop(
            columns=cols_to_drop,
            errors="ignore"
        )

    y = df[target_col]

    # Detect task type
    is_regression = (
        pd.api.types.is_numeric_dtype(y)
        and y.nunique() > 15
    )

    task_type = (
        "REGRESSION"
        if is_regression
        else "CLASSIFICATION"
    )

    print(
        f"\n[DETECTED TASK] {task_type} "
        f"based on target '{target_col}'"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    mlflow.set_experiment(experiment_name)

    # --------------------------------------------------------
    # OPTUNA OBJECTIVE
    # --------------------------------------------------------

    def objective(trial):

        with mlflow.start_run(nested=True):

            if is_regression:

                classifier_name = trial.suggest_categorical(
                    "model",
                    [
                        "LinearReg",
                        "RandomForest",
                        "XGBoost",
                        "SVR",
                        "KNN",
                        "LightGBM"
                    ]
                )

            else:

                classifier_name = trial.suggest_categorical(
                    "model",
                    [
                        "LogReg",
                        "RandomForest",
                        "XGBoost",
                        "SVM",
                        "KNN",
                        "LightGBM"
                    ]
                )

            params = {}

            if classifier_name == "LogReg":

                params = {
                    "logreg_c": trial.suggest_float(
                        "logreg_c",
                        1e-3,
                        10.0,
                        log=True
                    )
                }

            elif classifier_name == "RandomForest":

                params = {
                    "rf_n_estimators": trial.suggest_int(
                        "rf_n_estimators",
                        50,
                        200
                    ),

                    "rf_max_depth": trial.suggest_int(
                        "rf_max_depth",
                        3,
                        15
                    )
                }

            elif classifier_name in ["SVM", "SVR"]:

                params = {
                    "svm_c": trial.suggest_float(
                        "svm_c",
                        1e-3,
                        10.0,
                        log=True
                    )
                }

            elif classifier_name == "KNN":

                params = {
                    "knn_n_neighbors": trial.suggest_int(
                        "knn_n_neighbors",
                        3,
                        15
                    )
                }

            elif classifier_name == "LightGBM":

                params = {
                    "lgbm_lr": trial.suggest_float(
                        "lgbm_lr",
                        0.01,
                        0.3,
                        log=True
                    ),

                    "lgbm_num_leaves": trial.suggest_int(
                        "lgbm_num_leaves",
                        10,
                        100
                    )
                }

            else:

                params = {
                    "xgb_lr": trial.suggest_float(
                        "xgb_lr",
                        0.01,
                        0.3,
                        log=True
                    ),

                    "xgb_max_depth": trial.suggest_int(
                        "xgb_max_depth",
                        3,
                        10
                    )
                }

            model = instantiate_model(
                classifier_name,
                params,
                is_regression
            )

            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            scoring_metric = (
                "r2"
                if is_regression
                else "accuracy"
            )

            score = cross_val_score(
                full_pipeline,
                X_train,
                y_train,
                cv=3,
                scoring=scoring_metric
            ).mean()

            mlflow.log_param(
                "model_type",
                classifier_name
            )

            mlflow.log_params(params)

            mlflow.log_metric(
                f"cv_{scoring_metric}",
                score
            )

            return score

    # --------------------------------------------------------
    # MASTER AUTOML RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="AutoML_Master_Search"
    ):

        optuna.logging.set_verbosity(
            optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="maximize"
        )

        study.optimize(
            objective,
            n_trials=n_trials
        )

        best_params = study.best_params.copy()

        clf_type = best_params.pop("model")

        best_model = instantiate_model(
            clf_type,
            best_params,
            is_regression
        )

        winning_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])

        winning_pipeline.fit(
            X_train,
            y_train
        )

        scoring_metric = (
            "R2_Score"
            if is_regression
            else "Accuracy"
        )

        test_score = winning_pipeline.score(
            X_test,
            y_test
        )

        # Log metrics
        mlflow.log_param(
            "winning_model",
            clf_type
        )

        mlflow.log_param(
            "task_type",
            task_type
        )

        mlflow.log_metric(
            "test_score",
            test_score
        )

        # Log metric under a descriptive name
        mlflow.log_metric(
            scoring_metric,
            test_score
        )

        # Register model
        model_name = (
            override_name
            if override_name
            else f"{experiment_name}_champion"
        )

        mlflow.sklearn.log_model(
            sk_model=winning_pipeline,
            name="best_pipeline",
            registered_model_name=model_name,
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )
        )

        print(
            f"\n[SUCCESS] Model ({clf_type}) "
            f"registered as '{model_name}' "
            f"with Test {scoring_metric}: {test_score:.4f}"
        )

        # Update job status
        if job_id is not None and job_status_dict is not None:

            job_status_dict[job_id]["message"] = (
                f"Training complete. "
                f"Model: {clf_type}, "
                f"Score: {test_score:.4f}"
            )

        return {
            "winner": clf_type,
            "test_score": test_score,
            "model_name": model_name,
            "metric_name": scoring_metric
        }


# ============================================================
# MODEL RETRAINING JOB
# ============================================================

def run_retraining_job(
    df_old,
    df_new,
    target_col,
    experiment_name,
    current_metric_name,
    n_trials,
    job_id,
    job_status_dict
):

    try:

        job_status_dict[job_id]["message"] = (
            "Combining old and new data..."
        )

        # ----------------------------------------------------
        # 1. COMBINE OLD + NEW DATA
        # ----------------------------------------------------

        df_combined = pd.concat(
            [df_old, df_new],
            ignore_index=True
        )

        # Save combined dataset
        os.makedirs("data", exist_ok=True)

        combined_path = os.path.join(
            "data",
            f"{experiment_name}_original.csv"
        )

        df_combined.to_csv(
            combined_path,
            index=False
        )

        job_status_dict[job_id]["message"] = (
            "Training candidate model..."
        )

        # ----------------------------------------------------
        # 2. TRAIN CANDIDATE MODEL
        # ----------------------------------------------------

        candidate_name = (
            f"{experiment_name}_candidate"
        )

        run_automl_job(
            df_combined,
            target_col,
            f"{experiment_name}_retrain_exp",
            n_trials,
            job_id,
            job_status_dict,
            override_name=candidate_name
        )

        # ----------------------------------------------------
        # 3. FETCH CANDIDATE MODEL
        # ----------------------------------------------------

        client = MlflowClient()

        candidate_versions = (
            client.search_model_versions(
                f"name='{candidate_name}'"
            )
        )

        if not candidate_versions:

            raise Exception(
                "Candidate model was not registered in MLflow."
            )

        latest_candidate = max(
            candidate_versions,
            key=lambda version: int(version.version)
        )

        candidate_run = client.get_run(
            latest_candidate.run_id
        )

        # Find the requested metric
        metric_key = current_metric_name

        if metric_key not in candidate_run.data.metrics:

            if "test_score" in candidate_run.data.metrics:

                metric_key = "test_score"

            elif "R2_Score" in candidate_run.data.metrics:

                metric_key = "R2_Score"

            elif "Accuracy" in candidate_run.data.metrics:

                metric_key = "Accuracy"

            else:

                available_metrics = (
                    candidate_run.data.metrics
                )

                if not available_metrics:

                    raise Exception(
                        "Candidate model has no logged metrics."
                    )

                metric_key = next(
                    iter(available_metrics)
                )

        candidate_score = (
            candidate_run.data.metrics[metric_key]
        )

        # ----------------------------------------------------
        # 4. FETCH CHAMPION MODEL
        # ----------------------------------------------------

        champion_name = (
            f"{experiment_name}_champion"
        )

        champion_versions = (
            client.search_model_versions(
                f"name='{champion_name}'"
            )
        )

        if not champion_versions:

            raise Exception(
                "Champion model does not exist. "
                "Train the initial model before retraining."
            )

        latest_champion = max(
            champion_versions,
            key=lambda version: int(version.version)
        )

        champion_run = client.get_run(
            latest_champion.run_id
        )

        champion_score = (
            champion_run.data.metrics.get(
                metric_key,
                champion_run.data.metrics.get(
                    "test_score",
                    0.0
                )
            )
        )

        job_status_dict[job_id]["message"] = (
            f"Quality Gate: "
            f"Champion={champion_score:.4f}, "
            f"Candidate={candidate_score:.4f}"
        )

        # ----------------------------------------------------
        # 5. QUALITY GATE
        # ----------------------------------------------------

        if candidate_score >= champion_score:

            decision = "PROMOTED"

            # Archive previous champion
            client.set_model_version_tag(
                champion_name,
                latest_champion.version,
                "status",
                "Archived"
            )

            # Register candidate source as new champion version
            new_champion = client.create_model_version(
                name=champion_name,
                source=latest_candidate.source,
                run_id=latest_candidate.run_id
            )

            client.set_model_version_tag(
                champion_name,
                new_champion.version,
                "status",
                "Champion"
            )

        else:

            decision = "REJECTED"

        # ----------------------------------------------------
        # 6. LOG QUALITY GATE DECISION
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name="Quality_Gate_Decision"
        ):

            mlflow.log_metric(
                "champion_score",
                champion_score
            )

            mlflow.log_metric(
                "candidate_score",
                candidate_score
            )

            mlflow.set_tag(
                "decision",
                decision
            )

            mlflow.set_tag(
                "metric_name",
                metric_key
            )

            mlflow.set_tag(
                "experiment_name",
                experiment_name
            )

            mlflow.set_tag(
                "job_id",
                str(job_id)
            )

        # ----------------------------------------------------
        # 7. UPDATE JOB STATUS
        # ----------------------------------------------------

        job_status_dict[job_id] = {
            "status": "completed",
            "message": (
                f"✅ {decision}! "
                f"Candidate ({candidate_score:.4f}) "
                f"vs Champion ({champion_score:.4f})."
            )
        }

        print(
            f"\n[RETRAINING COMPLETE] {decision}"
        )

        print(
            f"Champion Score: {champion_score:.4f}"
        )

        print(
            f"Candidate Score: {candidate_score:.4f}"
        )

        print(
            f"Metric: {metric_key}"
        )

        return {
            "decision": decision,
            "champion_score": champion_score,
            "candidate_score": candidate_score,
            "metric_name": metric_key
        }

    except Exception as e:

        job_status_dict[job_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}"
        }

        print(
            f"\n[RETRAINING ERROR] {str(e)}"
        )

        return {
            "decision": "FAILED",
            "error": str(e)
        }
