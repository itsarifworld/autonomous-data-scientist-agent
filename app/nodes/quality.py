from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import DataProfile, QualityIssues

@node
def assess_quality(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Assesses data quality based on the data profile.
    """
    dataset_path = node_input["dataset_path"]
    profile_dict = node_input["profile"]
    charts = node_input["charts"]
    
    profile = DataProfile(**profile_dict)
    
    # High missing values (> 30%)
    total_rows = profile.metadata.rows
    high_missing = []
    for col, missing_count in profile.missing_values.items():
        if missing_count / total_rows > 0.3:
            high_missing.append(f"'{col}' has more than 30% missing values ({missing_count}/{total_rows}). Consider dropping it or using advanced imputation.")
            
    # Multicollinearity (> 0.8 correlation)
    multicollinearity = set()
    for col1, corrs in profile.correlations.items():
        for col2, corr_val in corrs.items():
            if col1 != col2 and abs(corr_val) > 0.8:
                # Add a sorted tuple to avoid duplicates like (A, B) and (B, A)
                pair = tuple(sorted([col1, col2]))
                multicollinearity.add(f"'{pair[0]}' and '{pair[1]}' are highly correlated ({corr_val:.2f}). One may be redundant for linear models.")
                
    # Basic Outlier detection using IQR from numerical stats
    outliers = {}
    for col, stats in profile.numerical_stats.items():
        if '25%' in stats and '75%' in stats:
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Since we don't have the full data here, we can only infer potential issues
            # if min/max are far outside the bounds
            if 'min' in stats and stats['min'] < lower_bound:
                outliers[col] = outliers.get(col, 0) + 1
            if 'max' in stats and stats['max'] > upper_bound:
                outliers[col] = outliers.get(col, 0) + 1
                
    quality_issues = QualityIssues(
        profile=profile,
        outliers=outliers,
        high_missing=high_missing,
        multicollinearity=list(multicollinearity)
    )
    
    output_data = {
        "dataset_path": dataset_path,
        "charts": charts,
        "quality_issues": quality_issues.model_dump()
    }
    return Event(output=output_data, state={"dataset_path": dataset_path, "charts": charts})
