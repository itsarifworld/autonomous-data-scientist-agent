# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
You can add your unit tests here.
This is where you test your business logic, including agent functionality,
data processing, and other core components of your application.
"""


from app.agent import app, root_agent
from app.utils.schemas import DataMetadata, ModelRecommendation


def test_app_configuration() -> None:
    """Test that ADK app and root workflow are properly configured."""
    assert app.name == "app"
    assert root_agent.name == "adsa_pipeline"
    assert len(root_agent.edges) == 12


def test_schemas() -> None:
    """Test data schemas and validation."""
    meta = DataMetadata(rows=100, columns=10, memory_usage_mb=1.5, dtypes={"col1": "int64"})
    assert meta.rows == 100
    rec = ModelRecommendation(model_name="Random Forest", justification="Robust model")
    assert rec.model_name == "Random Forest"
