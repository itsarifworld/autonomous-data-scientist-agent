import pandas as pd
import os
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import DataMetadata

import shlex

@node
def load_data(ctx: Context, node_input: str) -> Event:
    """
    Loads data from the given path (node_input) and computes metadata.
    Accepts format: "path/to/dataset.csv" or "path/to/dataset.csv --target price"
    """
    parts = shlex.split(node_input.strip())
    dataset_path = parts[0]
    
    explicit_target = None
    if "--target" in parts:
        idx = parts.index("--target")
        if idx + 1 < len(parts):
            explicit_target = parts[idx + 1]
            
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    ctx.state["dataset_path"] = dataset_path
    if explicit_target:
        ctx.state["explicit_target"] = explicit_target
        
    # Read dataset
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
        
    # Compute metadata
    metadata = DataMetadata(
        rows=len(df),
        columns=len(df.columns),
        memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()}
    )
    
    output_data = {
        "dataset_path": dataset_path,
        "metadata": metadata.model_dump()
    }
    
    return Event(output=output_data)
