import pandas as pd
from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import DataProfile, DataMetadata
from app.utils.visualize import generate_all_charts

@node
def profile_data(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Profiles the dataset and generates charts.
    """
    dataset_path = node_input["dataset_path"]
    metadata_dict = node_input["metadata"]
    
    # Read dataset
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
    
    # Missing values
    missing_values = df.isnull().sum().to_dict()
    
    # Numerical stats
    num_df = df.select_dtypes(include=['number'])
    numerical_stats = {}
    if not num_df.empty:
        desc = num_df.describe().to_dict()
        for col, stats in desc.items():
            numerical_stats[col] = {k: float(v) for k, v in stats.items()}
            
    # Categorical counts
    cat_df = df.select_dtypes(exclude=['number'])
    categorical_counts = {}
    for col in cat_df.columns:
        categorical_counts[col] = int(cat_df[col].nunique())
        
    # Correlations
    correlations = {}
    if not num_df.empty and len(num_df.columns) > 1:
        corr_matrix = num_df.corr().to_dict()
        for col, corrs in corr_matrix.items():
            correlations[col] = {k: float(v) if pd.notnull(v) else 0.0 for k, v in corrs.items()}
            
    # Generate Charts
    charts = generate_all_charts(df, output_dir="outputs")
    
    profile = DataProfile(
        metadata=DataMetadata(**metadata_dict),
        missing_values=missing_values,
        numerical_stats=numerical_stats,
        categorical_counts=categorical_counts,
        correlations=correlations
    )
    
    output_data = {
        "dataset_path": dataset_path,
        "profile": profile.model_dump(),
        "charts": charts
    }
    
    return Event(output=output_data)
