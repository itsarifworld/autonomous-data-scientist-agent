from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import ModelRecommendation

@node
def recommend_models(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Heuristically recommends ML models based on the dataset profile and detected problem type.
    """
    problem = node_input["problem_detection"]
    quality = node_input["quality_issues"]
    profile = quality["profile"]
    
    task_type = problem["task_type"]
    rows = profile["metadata"]["rows"]
    has_missing = len(quality["high_missing"]) > 0 or any(v > 0 for v in profile["missing_values"].values())
    has_multicollinearity = len(quality["multicollinearity"]) > 0
    
    recommendations = []
    
    if task_type == "Clustering":
        recommendations.append(ModelRecommendation(
            model_name="K-Means Clustering",
            justification="Standard baseline for clustering tasks. Works well with continuous numerical data."
        ))
        recommendations.append(ModelRecommendation(
            model_name="DBSCAN",
            justification="Recommended if the data has irregular shapes or many outliers, as it does not assume spherical clusters."
        ))
        recommendations.append(ModelRecommendation(
            model_name="Isolation Forest",
            justification="Excellent for anomaly detection if the goal is to find rare, unusual records."
        ))
    else:
        # Supervised Learning
        if rows < 100:
            # Small dataset
            if task_type == "Classification":
                recommendations.append(ModelRecommendation(
                    model_name="Logistic Regression (with Regularization)",
                    justification="Highly interpretable and less prone to overfitting on very small datasets."
                ))
                recommendations.append(ModelRecommendation(
                    model_name="Decision Tree",
                    justification="Simple, interpretable, and handles mixed data types without scaling."
                ))
            else:
                recommendations.append(ModelRecommendation(
                    model_name="Ridge/Lasso Regression",
                    justification="Regularized linear models prevent extreme overfitting on small datasets."
                ))
                recommendations.append(ModelRecommendation(
                    model_name="K-Nearest Neighbors (KNN)",
                    justification="Non-parametric approach that works well when relationships are highly non-linear in small dimensions."
                ))
                
        else:
            # Normal/Large dataset
            if has_missing:
                recommendations.append(ModelRecommendation(
                    model_name="XGBoost",
                    justification="State-of-the-art performance and natively handles missing values without requiring explicit imputation."
                ))
                recommendations.append(ModelRecommendation(
                    model_name="LightGBM",
                    justification="Extremely fast, handles missing values natively, and excellent for large datasets."
                ))
            else:
                recommendations.append(ModelRecommendation(
                    model_name="Random Forest",
                    justification="Robust, handles mixed data types, and less prone to overfitting than pure decision trees."
                ))
                
            if has_multicollinearity:
                recommendations.append(ModelRecommendation(
                    model_name="Ridge Regression" if task_type == "Regression" else "Logistic Regression (L2 Penalty)",
                    justification="Penalized linear models are required to handle the severe multicollinearity detected in the dataset."
                ))
            else:
                recommendations.append(ModelRecommendation(
                    model_name="Linear Regression" if task_type == "Regression" else "Logistic Regression",
                    justification="Strong baseline model given the lack of severe multicollinearity."
                ))
                
    # Take top 3
    node_input["model_recommendations"] = [r.model_dump() for r in recommendations[:3]]
    return Event(output=node_input)
