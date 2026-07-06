import os
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from app.utils.schemas import InsightResponse

# Read prompt
prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "insight_prompt.md")
with open(prompt_path, "r") as f:
    instruction = f.read()

insight_agent = LlmAgent(
    name="insight_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction=instruction,
    output_schema=InsightResponse,
    output_key="insights"
)
