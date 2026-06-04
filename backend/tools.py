from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool

from backend.analysis import build_analysis_plan
from backend.data import dataframe_schema, execute_sql_on_dataframe, sanitize_sql, summarize_result


def get_tools(df: pd.DataFrame) -> list:
    @tool
    def inspect_dataset_schema() -> str:
        """Inspect the current business dataset schema and sample rows."""
        return dataframe_schema(df)

    @tool
    def analyze_business_data(
        question: str,
        group_by: str = "",
        metric: str = "",
        date_grain: str = "",
        chart_type: str = "bar",
    ) -> str:
        """Analyze business data by intent without requiring the model to write SQL."""
        plan = build_analysis_plan(
            question=question,
            df=df,
            group_by=group_by,
            metric=metric,
            date_grain=date_grain,
            chart_type=chart_type,
        )
        safe_sql = sanitize_sql(plan["sql"])
        result_df = execute_sql_on_dataframe(df, safe_sql)
        summary = summarize_result(result_df, df)
        payload = {
            "intent": plan["intent"],
            "sql": safe_sql,
            "chart_type": plan["chart_type"],
            "chart_x": plan["chart_x"],
            "chart_y": plan["chart_y"],
            "chart_color": plan["chart_color"],
            "analysis": {
                "rows_returned": summary.rows_returned,
                "total_revenue": summary.total_revenue,
                "total_profit": summary.total_profit,
                "profit_margin": summary.profit_margin,
                "top_dimension": summary.top_dimension,
                "weak_dimension": summary.weak_dimension,
            },
            "rows": result_df.to_dict(orient="records"),
        }
        return json.dumps(payload, default=str)

    return [inspect_dataset_schema, analyze_business_data]
