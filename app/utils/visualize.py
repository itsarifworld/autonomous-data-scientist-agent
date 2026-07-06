import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from typing import Any

def plot_missing_values(df: pd.DataFrame, output_dir: str) -> str:
    """Plots a bar chart of missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        return ""
        
    fig = px.bar(
        x=missing.index, 
        y=missing.values,
        labels={'x': 'Columns', 'y': 'Missing Count'},
        title="Missing Values per Column",
        template="plotly_dark"
    )
    
    path = os.path.join(output_dir, "missing_values.html")
    fig.write_html(path)
    return path

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Plots a correlation heatmap for numerical features."""
    num_df = df.select_dtypes(include=['number'])
    
    if num_df.empty or len(num_df.columns) < 2:
        return ""
        
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto",
        title="Numerical Features Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        template="plotly_dark"
    )
    
    path = os.path.join(output_dir, "correlation_heatmap.html")
    fig.write_html(path)
    return path

def plot_numerical_distributions(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Plots histograms for numerical distributions."""
    num_cols = df.select_dtypes(include=['number']).columns
    paths = []
    
    # Limit to top 10 features to avoid overwhelming the output
    for col in num_cols[:10]:
        fig = px.histogram(
            df, 
            x=col, 
            title=f"Distribution of {col}",
            template="plotly_dark",
            marginal="box"
        )
        path = os.path.join(output_dir, f"dist_{col}.html")
        fig.write_html(path)
        paths.append(path)
        
    return paths

def generate_all_charts(df: pd.DataFrame, output_dir: str) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    return {
        "missing_values": plot_missing_values(df, output_dir),
        "correlation": plot_correlation_heatmap(df, output_dir),
        "distributions": plot_numerical_distributions(df, output_dir)
    }
