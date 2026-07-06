from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

def get_model_instance(model_name: str, task_type: str):
    """Maps the string model name to a scikit-learn instance."""
    name = model_name.lower()
    
    if task_type == "Clustering":
        if "k-means" in name: return KMeans(n_clusters=3, random_state=42)
        if "dbscan" in name: return DBSCAN()
        if "isolation" in name: return IsolationForest(random_state=42)
        return KMeans(n_clusters=3, random_state=42) # default
        
    is_class = task_type == "Classification"
    
    if "logistic" in name: return LogisticRegression(max_iter=1000, random_state=42)
    if "linear" in name: return LinearRegression()
    if "ridge" in name or "lasso" in name: return Ridge(random_state=42)
    if "decision tree" in name: 
        return DecisionTreeClassifier(random_state=42) if is_class else DecisionTreeRegressor(random_state=42)
    if "knn" in name or "k-nearest" in name:
        return KNeighborsClassifier() if is_class else KNeighborsRegressor()
    if "random forest" in name:
        return RandomForestClassifier(random_state=42) if is_class else RandomForestRegressor(random_state=42)
    if "xgboost" in name or "lightgbm" in name:
        # Fallback to sklearn's native histogram-based gradient boosting (equivalent to LightGBM)
        return HistGradientBoostingClassifier(random_state=42) if is_class else HistGradientBoostingRegressor(random_state=42)
        
    # Default fallback
    return RandomForestClassifier(random_state=42) if is_class else RandomForestRegressor(random_state=42)


@node
def train_models(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Splits the data and trains all recommended models.
    """
    import pandas as pd
    import os
    
    X_path = ctx.state.get("X_cleaned_path")
    y_path = ctx.state.get("y_cleaned_path")
    
    if not X_path or not y_path:
        raise ValueError("Cleaned data paths not found in state.")
        
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path).iloc[:, 0]
    
    problem = node_input.get("problem_detection", {})
    task_type = problem.get("task_type", "Classification")
    
    # Train Test Split
    if task_type != "Clustering":
        # For small datasets, stratify if classification to ensure both classes exist
        stratify_col = y if task_type == "Classification" and len(X) >= 10 else None
        
        # If the dataset is too small or classes are too imbalanced to stratify, turn it off.
        if stratify_col is not None:
            class_counts = y.value_counts()
            if any(class_counts < 2):
                stratify_col = None
                
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_col)
        except ValueError:
            # Fallback if stratify fails
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y # Clustering uses all data
        
    os.makedirs("outputs", exist_ok=True)
    X_train.to_parquet("outputs/X_train.parquet")
    X_test.to_parquet("outputs/X_test.parquet")
    pd.DataFrame(y_train).to_parquet("outputs/y_train.parquet")
    pd.DataFrame(y_test).to_parquet("outputs/y_test.parquet")
    
    ctx.state["X_train_path"] = "outputs/X_train.parquet"
    ctx.state["X_test_path"] = "outputs/X_test.parquet"
    ctx.state["y_train_path"] = "outputs/y_train.parquet"
    ctx.state["y_test_path"] = "outputs/y_test.parquet"
    
    trained_models = {}
    recommendations = node_input.get("model_recommendations", [])
    
    for rec in recommendations:
        model_name = rec["model_name"]
        model = get_model_instance(model_name, task_type)
        
        try:
            if task_type == "Clustering" or model_name == "Isolation Forest":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)
            trained_models[model_name] = model
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            
    # Save models using joblib to files instead of storing them directly in state
    model_paths = {}
    for name, model in trained_models.items():
        # Sanitize name for file path
        safe_name = "".join([c if c.isalnum() else "_" for c in name])
        path = f"outputs/model_{safe_name}.pkl"
        import joblib
        joblib.dump(model, path)
        model_paths[name] = path
        
    ctx.state["trained_model_paths"] = model_paths
    
    return Event(output=node_input)
