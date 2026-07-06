from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class DataMetadata(BaseModel):
    rows: int
    columns: int
    memory_usage_mb: float
    dtypes: Dict[str, str]

class DataProfile(BaseModel):
    metadata: DataMetadata
    missing_values: Dict[str, int]
    numerical_stats: Dict[str, Dict[str, float]]
    categorical_counts: Dict[str, int]
    correlations: Dict[str, Dict[str, float]]

class QualityIssues(BaseModel):
    profile: DataProfile
    outliers: Dict[str, int]
    high_missing: List[str]
    multicollinearity: List[str]

class StatisticalTestResult(BaseModel):
    test_name: str
    feature_a: str
    feature_b: Optional[str] = None
    p_value: float
    is_significant: bool
    conclusion: str

class ProblemDetection(BaseModel):
    task_type: str
    potential_targets: List[str]
    reasoning: str

class ModelRecommendation(BaseModel):
    model_name: str
    justification: str

class AutoMLResult(BaseModel):
    best_model_name: str
    target_variable: str
    test_score: float
    metric_name: str
    top_features_shap: List[str]

class AdvancedAnalytics(BaseModel):
    quality_issues: QualityIssues
    statistical_tests: List[StatisticalTestResult]
    problem_detection: ProblemDetection
    model_recommendations: List[ModelRecommendation]
    automl_results: AutoMLResult

class InsightResponse(BaseModel):
    dataset_overview: str
    data_quality_assessment: str
    statistical_findings: str
    machine_learning_task_recommendation: str
    feature_engineering_recommendations: List[str]
    model_recommendations_summary: str
    automl_evaluation_summary: str
    priority_actions: List[str]

class FinalReport(BaseModel):
    report_path: str
    charts_dir: str
