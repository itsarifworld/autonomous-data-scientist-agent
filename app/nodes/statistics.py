import pandas as pd
from typing import Any, Dict
from scipy import stats
from google.adk.workflow import node
from google.adk.agents.context import Context
from google.adk.events.event import Event
from app.utils.schemas import StatisticalTestResult
import warnings

@node
def run_statistics(ctx: Context, node_input: Dict[str, Any]) -> Event:
    """
    Runs statistical tests (Normality, Chi-Square, T-Test/ANOVA) on the dataset.
    """
    dataset_path = node_input["dataset_path"]
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_parquet(dataset_path)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    test_results = []
    
    # Suppress scipy warnings for small datasets in tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 1. Normality Tests (Shapiro-Wilk)
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) >= 3:
                stat, p = stats.shapiro(col_data)
                is_sig = p < 0.05
                test_results.append(
                    StatisticalTestResult(
                        test_name="Shapiro-Wilk Normality",
                        feature_a=col,
                        p_value=p,
                        is_significant=is_sig,
                        conclusion=f"'{col}' is {'NOT normally distributed' if is_sig else 'normally distributed'}."
                    )
                )

        # 2. Chi-Square (Categorical vs Categorical)
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1 = categorical_cols[i]
                col2 = categorical_cols[j]
                contingency_table = pd.crosstab(df[col1], df[col2])
                if contingency_table.size > 0:
                    stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                    is_sig = p < 0.05
                    test_results.append(
                        StatisticalTestResult(
                            test_name="Chi-Square Independence",
                            feature_a=col1,
                            feature_b=col2,
                            p_value=p,
                            is_significant=is_sig,
                            conclusion=f"'{col1}' and '{col2}' are {'significantly associated' if is_sig else 'independent'}."
                        )
                    )

        # 3. T-Test/ANOVA (Categorical vs Numerical)
        for cat_col in categorical_cols:
            unique_vals = df[cat_col].dropna().unique()
            if len(unique_vals) == 2:
                # T-Test
                group1 = df[df[cat_col] == unique_vals[0]]
                group2 = df[df[cat_col] == unique_vals[1]]
                for num_col in numerical_cols:
                    g1_data = group1[num_col].dropna()
                    g2_data = group2[num_col].dropna()
                    if len(g1_data) > 1 and len(g2_data) > 1:
                        stat, p = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                        is_sig = p < 0.05
                        test_results.append(
                            StatisticalTestResult(
                                test_name="Welch's T-Test",
                                feature_a=cat_col,
                                feature_b=num_col,
                                p_value=p,
                                is_significant=is_sig,
                                conclusion=f"Significant difference in '{num_col}' across '{cat_col}'" if is_sig else f"No significant difference in '{num_col}' across '{cat_col}'"
                            )
                        )
            elif 2 < len(unique_vals) <= 10:
                # ANOVA
                for num_col in numerical_cols:
                    groups = [df[df[cat_col] == val][num_col].dropna() for val in unique_vals]
                    groups = [g for g in groups if len(g) > 1]
                    if len(groups) > 1:
                        stat, p = stats.f_oneway(*groups)
                        is_sig = p < 0.05
                        test_results.append(
                            StatisticalTestResult(
                                test_name="One-Way ANOVA",
                                feature_a=cat_col,
                                feature_b=num_col,
                                p_value=p,
                                is_significant=is_sig,
                                conclusion=f"Significant difference in '{num_col}' across '{cat_col}' groups" if is_sig else f"No significant difference in '{num_col}' across '{cat_col}' groups"
                            )
                        )
                        
    # Append to output payload
    node_input["statistical_tests"] = [r.model_dump() for r in test_results]
    return Event(output=node_input)
