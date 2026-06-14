import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_log_model(data: pd.DataFrame):
    """Trains a pipeline model and logs it to MLflow."""
    X = data.drop(columns=['failure'])
    y = data['failure']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create Pipeline (Scaler + Model)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        ))
    ])

    mlflow.set_experiment('predictive-maintenance')

    with mlflow.start_run(run_name=f"auto_retrain_{datetime.now().strftime('%Y%m%d')}"):
        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred_proba)

        # Log
        mlflow.log_params({'n_estimators': 100, 'max_depth': 6})
        mlflow.log_metrics({'accuracy': acc, 'roc_auc': roc})
        mlflow.log_param('retrain_date', datetime.now().isoformat())
        
        # Log Pipeline
        mlflow.sklearn.log_model(
            pipeline, 
            artifact_path='model', 
            registered_model_name='PredictiveMaintenance'
        )

        logger.info(f"Model trained. Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}")
        return roc

if __name__ == "__main__":
    # Assuming generate_synthetic_data() exists from previous step
    logger.info('Running retraining pipeline...')
    new_data = generate_synthetic_data() 
    train_and_log_model(new_data)