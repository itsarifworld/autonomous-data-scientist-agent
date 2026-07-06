from google.adk.workflow import Workflow
from app.nodes.loading import load_data
from app.nodes.profiling import profile_data
from app.nodes.quality import assess_quality
from app.nodes.statistics import run_statistics
from app.nodes.problem_detection import identify_problem
from app.nodes.model_recommender import recommend_models
from app.nodes.cleaning import clean_data
from app.nodes.training import train_models
from app.nodes.evaluation import evaluate_models
from app.nodes.shap_analysis import shap_analysis
from app.agents.insight_agent import insight_agent
from app.nodes.report import generate_report

root_agent = Workflow(
    name="adsa_pipeline",
    description="Autonomous Data Scientist Agent - Phase 4 AutoML",
    edges=[
        ('START', load_data),
        (load_data, profile_data),
        (profile_data, assess_quality),
        (assess_quality, run_statistics),
        (run_statistics, identify_problem),
        (identify_problem, recommend_models),
        (recommend_models, clean_data),
        (clean_data, train_models),
        (train_models, evaluate_models),
        (evaluate_models, shap_analysis),
        (shap_analysis, insight_agent),
        (insight_agent, generate_report)
    ]
)
