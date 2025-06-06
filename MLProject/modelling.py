import os
import sys
import io
import pandas as pd
import mlflow
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score 
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
import numpy as np

def modeling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path):
    # Load dataset
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
    
    print(f"Jumlah data X_train: {X_train.shape[0]}")
    print(f"Jumlah data X_test: {X_test.shape[0]}")
    print(f"Jumlah data y_train: {len(y_train)}")
    print(f"Jumlah data y_test: {len(y_test)}")

    num_classes = y_train.nunique()
    print(f"Jumlah kelas unik: {num_classes}")

    # Fungsi objective Optuna untuk Tuning disertai Cross Validation
    def objective(trial):
        params = {
            "objective": "multi:softmax",
            "num_class": num_classes,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss"
        }
        model = xgb.XGBClassifier(**params)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score, average='macro')
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring=scorer, n_jobs=-1)
        mean_score = np.mean(cv_scores)
        
        # Logging per trial tuning
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("f1_macro", mean_score)
            
        return mean_score

    print("Hyperparameter tuning dengan Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    best_params.update({
        "objective": "multi:softmax",
        "num_class": num_classes,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss"
    })
    print("Param terbaik:", best_params)

    # Final training model Menggunakan Parameter Terbaik
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # MLFlow Logging Manual
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
    mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

    print("\nEvaluasi Model:")
    print("Akurasi:", acc)
    print(classification_report(y_test, y_pred))
    
    # Logging model sebagai artifact ke MLflow (DagsHub)
    mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

    return model

if __name__ == "__main__":
    # Path file hasil preprocessing dan split
    X_train_path = "data_balita_preprocessing/X_train.csv"
    X_test_path = "data_balita_preprocessing/X_test.csv"
    y_train_path = "data_balita_preprocessing/y_train.csv"
    y_test_path = "data_balita_preprocessing/y_test.csv"

    # Inisialisasi MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = "nandaarya"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "75280f51e98b7a881cc6ba32ba12267d27c1f20e"
    
    mlflow.set_tracking_uri("https://dagshub.com/nandaarya/stunting-detection.mlflow")
    mlflow.set_experiment("Stunting_Detection_XGBoost_Model_Tuning")
    
    # Atasi UnicodeEncodeError saat mencetak karakter seperti emoji
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    with mlflow.start_run(run_name="XGBoost_with_Optuna"):
        trained_model = modeling_with_tuning(X_train_path, X_test_path,y_train_path, y_test_path)