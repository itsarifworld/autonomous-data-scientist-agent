from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
import os
import joblib
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import numpy as np

@node
def evaluate_models(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Evaluates trained models, selects the best one, and saves it to disk.
    """
    import pandas as pd
    model_paths = ctx.state.get("trained_model_paths", {})
    if not model_paths:
        raise ValueError("No trained models found in state.")
        
    X_test = pd.read_parquet(ctx.state["X_test_path"])
    y_test = pd.read_parquet(ctx.state["y_test_path"]).iloc[:, 0]
    X_train = pd.read_parquet(ctx.state["X_train_path"]) # for clustering
    target_col = ctx.state.get("target_col", "Unknown Target")
    
    problem = node_input.get("problem_detection", {})
    task_type = problem.get("task_type", "Classification")
    
    best_model_name = ""
    best_score = -float('inf')
    best_model = None
    metric_name = "Accuracy" if task_type == "Classification" else ("Negative RMSE" if task_type == "Regression" else "Silhouette Score")
    
    for name, path in model_paths.items():
        try:
            model = joblib.load(path)
            if task_type == "Clustering":
                labels = model.fit_predict(X_train)
                # Silhouette score needs at least 2 clusters
                if len(set(labels)) > 1:
                    score = silhouette_score(X_train, labels)
                else:
                    score = -1
            else:
                y_pred = model.predict(X_test)
                if task_type == "Classification":
                    score = accuracy_score(y_test, y_pred)
                else:
                    # For regression, higher is better so we use negative RMSE
                    score = -np.sqrt(mean_squared_error(y_test, y_pred))
                    
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = model
                
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            
    if not best_model:
        raise RuntimeError("No models were successfully evaluated.")
        
    # Save best model
    os.makedirs("outputs", exist_ok=True)
    best_path = "outputs/best_model.pkl"
    joblib.dump(best_model, best_path)
    ctx.state["best_model_name"] = best_model_name
    
    # We use negative RMSE internally for sorting, but let's report positive RMSE if Regression
    reported_score = -best_score if task_type == "Regression" else best_score
    reported_metric = "RMSE" if task_type == "Regression" else metric_name
    
    # Prepare partial AutoML result (SHAP will fill in top features later)
    node_input["automl_results"] = {
        "best_model_name": best_model_name,
        "target_variable": target_col,
        "test_score": float(reported_score),
        "metric_name": reported_metric,
        "top_features_shap": [] # To be filled
    }
    
    return Event(output=node_input)
