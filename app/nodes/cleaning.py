from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

@node
def clean_data(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Cleans the dataset by handling missing values and encoding categorical features.
    Auto-selects the target variable based on problem_detection.
    """
    # Load raw dataset path
    dataset_path = ctx.state.get("dataset_path")
    if not dataset_path:
        raise ValueError("Dataset path not found in state.")
        
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
    
    # Auto-select Target
    problem = node_input.get("problem_detection", {})
    potential_targets = problem.get("potential_targets", [])
    explicit_target = ctx.state.get("explicit_target")
    
    if explicit_target:
        target_col = explicit_target
    elif potential_targets:
        target_col = potential_targets[0]
    else:
        # Fallback to the last column
        target_col = df.columns[-1]
        
    # Drop target if it has missing values (we can't train on missing targets easily)
    df = df.dropna(subset=[target_col])
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Identify column types
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(exclude=['number']).columns
    
    # Impute missing values
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
        # One-hot encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
        
        # Combine
        X = pd.concat([X[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # Save processed data to state as file paths
    X.to_parquet("outputs/X_cleaned.parquet")
    pd.DataFrame(y).to_parquet("outputs/y_cleaned.parquet")
    
    ctx.state["X_cleaned_path"] = "outputs/X_cleaned.parquet"
    ctx.state["y_cleaned_path"] = "outputs/y_cleaned.parquet"
    ctx.state["target_col"] = target_col
    
    return Event(output=node_input)
