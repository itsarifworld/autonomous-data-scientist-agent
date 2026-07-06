from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

@node
def shap_analysis(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Computes SHAP values for the best model to determine global feature importance.
    """
    automl_results = node_input.get("automl_results")
    if not automl_results:
        return Event(output=node_input)
        
    best_model_name = automl_results["best_model_name"]
    X_train = pd.read_parquet(ctx.state["X_train_path"])
    
    # SHAP can be computationally expensive and model-dependent. 
    # Wrap in try/except to avoid crashing the pipeline if explainer fails.
    try:
        model = joblib.load("outputs/best_model.pkl")
        
        # Determine explainer type heuristically
        if "Forest" in best_model_name or "Tree" in best_model_name or "Boost" in best_model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        elif "Linear" in best_model_name or "Logistic" in best_model_name or "Ridge" in best_model_name or "Lasso" in best_model_name:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        else:
            # Fallback for KNN, etc. Can be extremely slow, so sample heavily.
            background = shap.sample(X_train, min(50, len(X_train)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_train.sample(min(100, len(X_train))))
            
        # Extract mean absolute SHAP values for global feature importance
        import numpy as np
        if isinstance(shap_values, list):
            # For multi-class classification, take the sum over classes
            mean_abs_shap = sum([np.abs(sv).mean(0) for sv in shap_values])
        else:
            mean_abs_shap = np.abs(shap_values).mean(0)
            
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': mean_abs_shap
        }).sort_values(by='importance', ascending=False)
        
        top_features = feature_importance['feature'].head(3).tolist()
        automl_results["top_features_shap"] = top_features
        
        # Save summary plot
        plt.figure(figsize=(10, 6))
        # Ensure we don't block the execution thread
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("outputs/shap_summary.png", bbox_inches='tight')
        plt.close()
        
        # Add shap plot path to charts state so report.py includes it
        charts = ctx.state.get("charts", {})
        charts["SHAP Feature Importance"] = "outputs/shap_summary.png"
        ctx.state["charts"] = charts
        
    except Exception as e:
        print(f"SHAP analysis failed for {best_model_name}: {e}")
        automl_results["top_features_shap"] = ["Unable to compute SHAP values for this model."]
        
    node_input["automl_results"] = automl_results
    return Event(output=node_input)
