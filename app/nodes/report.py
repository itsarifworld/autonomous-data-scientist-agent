from typing import Any, Dict
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import FinalReport, InsightResponse
import os
import json

@node
def generate_report(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Takes the output from the insight_agent (which is a dict matching InsightResponse)
    and the charts/path from state, and generates a markdown report.
    """
    # LlmAgent with output_schema returns a dict matching the schema.
    insights = InsightResponse(**node_input)
    
    # We need the charts and dataset_path from earlier nodes. 
    # Since LlmAgent only outputs its own generation, we must retrieve them from state.
    # We need to make sure the quality node saves these to state!
    # Wait, if LlmAgent is a node, its input is the predecessor's output.
    # LlmAgent does NOT pass through the predecessor's output. 
    # So we must read from ctx.state.
    
    dataset_path = ctx.state.get("dataset_path", "Unknown Dataset")
    charts = ctx.state.get("charts", {})
    
    report_content = f"# Autonomous Data Scientist Report\n\n"
    report_content += f"**Dataset:** `{dataset_path}`\n\n"
    
    report_content += "## 1. Dataset Overview\n"
    report_content += f"{insights.dataset_overview}\n\n"
    
    report_content += "## 2. Data Quality Assessment\n"
    report_content += f"{insights.data_quality_assessment}\n\n"
    
    report_content += "## 3. Statistical Findings\n"
    report_content += f"{insights.statistical_findings}\n\n"
    
    report_content += "## 4. Machine Learning Task Recommendation\n"
    report_content += f"{insights.machine_learning_task_recommendation}\n\n"
    
    report_content += "## 5. Feature Engineering Recommendations\n"
    for rec in insights.feature_engineering_recommendations:
        report_content += f"- {rec}\n"
    report_content += "\n"
    
    report_content += "## 6. Recommended Machine Learning Models\n"
    report_content += f"{insights.model_recommendations_summary}\n\n"
    
    report_content += "## 7. Model Training & Evaluation (AutoML)\n"
    report_content += f"{insights.automl_evaluation_summary}\n\n"
    
    report_content += "## 8. Priority Actions\n"
    for action in insights.priority_actions:
        report_content += f"- {action}\n"
    report_content += "\n"
    
    report_content += "## Generated Charts\n"
    report_content += "Interactive charts have been saved to the `outputs/` directory:\n"
    for name, path in charts.items():
        if isinstance(path, list):
            for p in path:
                report_content += f"- [Distribution]({p})\n"
        else:
            if path:
                report_content += f"- [{name}]({path})\n"
                
    report_path = "outputs/final_report.md"
    os.makedirs("outputs", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_content)
        
    final_output = FinalReport(report_path=report_path, charts_dir="outputs")
    
    from google.genai import types
    
    # Yield the markdown string so it streams to the CLI / playground
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=report_content)])
    )
    
    # Yield the final output to terminate the node properly
    yield Event(output=report_content)
