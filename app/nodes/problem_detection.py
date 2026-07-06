from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import ProblemDetection

@node
def identify_problem(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Heuristically identifies whether the ML task should be Classification, Regression, or Clustering.
    """
    profile_dict = node_input["quality_issues"]["profile"]
    
    categorical_counts = profile_dict["categorical_counts"]
    numerical_stats = profile_dict["numerical_stats"]
    
    # Identify potential target columns
    potential_classification_targets = [col for col, count in categorical_counts.items() if 2 <= count <= 10]
    
    # Filter out obvious ID columns for regression
    potential_regression_targets = []
    for col, stats in numerical_stats.items():
        if col.lower() not in ['id', 'index', 'key']:
            potential_regression_targets.append(col)
            
    task_type = "Clustering"
    reasoning = "No obvious target variable detected. Clustering or anomaly detection is recommended."
    targets = []
    
    explicit_target = ctx.state.get("explicit_target")
    
    if explicit_target:
        if explicit_target in categorical_counts:
            task_type = "Classification"
        else:
            task_type = "Regression"
        targets = [explicit_target]
        reasoning = f"User explicitly provided target variable: {explicit_target}. Determined task type as {task_type}."
    elif potential_classification_targets:
        task_type = "Classification"
        targets = potential_classification_targets
        reasoning = f"Detected categorical columns with few unique values ({', '.join(targets)}), suitable for Classification target variables."
    elif potential_regression_targets:
        task_type = "Regression"
        targets = potential_regression_targets
        reasoning = f"No obvious classification targets found, but numerical columns exist ({', '.join(targets[:3])}...), suitable for Regression."
        
    problem_detection = ProblemDetection(
        task_type=task_type,
        potential_targets=targets,
        reasoning=reasoning
    )
    
    node_input["problem_detection"] = problem_detection.model_dump()
    return Event(output=node_input)
