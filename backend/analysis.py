from __future__ import annotations

import pandas as pd

from backend.config import TABLE_NAME


def wants_time_series(question: str) -> bool:
    q = question.lower()
    return any(
        phrase in q
        for phrase in [
            "trend",
            "trends",
            "monthly",
            "over time",
            "by month",
            "linear chart",
            "line chart",
            "across the 3 months",
            "across 3 months",
            "across the three months",
            "across three months",
        ]
    )


def pick_metric(question: str, df: pd.DataFrame, metric_hint: str = "") -> str:
    numeric_cols = list(df.select_dtypes("number").columns)
    q = f"{question} {metric_hint}".lower()
    if any(phrase in q for phrase in ["products sold", "product sold", "items sold", "units sold", "sold in"]):
        if "units_sold" in df.columns:
            return "units_sold"
    for candidate in ["profit", "revenue", "cost", "units_sold", "unit_price"]:
        if candidate in df.columns and candidate.replace("_", " ") in q:
            return candidate
    if "sales" in q and "revenue" in df.columns:
        return "revenue"
    if metric_hint in df.columns:
        return metric_hint
    if "revenue" in df.columns:
        return "revenue"
    return numeric_cols[0] if numeric_cols else df.columns[0]


def infer_group_columns(question: str, df: pd.DataFrame, group_hint: str = "") -> list[str]:
    q = f"{question} {group_hint}".lower()
    groups: list[str] = []

    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col and wants_time_series(q):
        groups.append(f"strftime('%Y-%m', {date_col}) AS month")

    for candidate in ["region", "product", "category"]:
        if (
            candidate == "product"
            and "region" in q
            and "underperform" in q
            and wants_time_series(q)
        ):
            continue
        if candidate in df.columns and candidate in q:
            groups.append(candidate)

    has_primary_breakdown = any(group in groups for group in ["region", "product", "category"])
    if "underperform" in q and "product" in df.columns and "product" not in groups and not has_primary_breakdown:
        groups.append("product")
    if not groups:
        for candidate in ["region", "product", "category"]:
            if candidate in df.columns:
                groups.append(candidate)
                break
    return groups


def infer_chart_type(question: str, chart_hint: str = "") -> str:
    q = f"{question} {chart_hint}".lower()
    if "pie" in q or "donut" in q:
        return "pie"
    if wants_time_series(q) or "line" in q or "linear" in q:
        return "line"
    if "scatter" in q:
        return "scatter"
    return chart_hint or "bar"


def infer_where_clause(question: str, df: pd.DataFrame) -> str:
    q = question.lower()
    date_col = next((c for c in df.columns if "date" in c), None)
    if not date_col:
        return ""

    if "q4" in q and "2025" in q:
        return f"WHERE {date_col} >= '2025-10-01' AND {date_col} < '2026-01-01'"
    if "q3" in q and "2025" in q:
        return f"WHERE {date_col} >= '2025-07-01' AND {date_col} < '2025-10-01'"
    if "q2" in q and "2025" in q:
        return f"WHERE {date_col} >= '2025-04-01' AND {date_col} < '2025-07-01'"
    if "q1" in q and "2025" in q:
        return f"WHERE {date_col} >= '2025-01-01' AND {date_col} < '2025-04-01'"
    return ""


def build_analysis_plan(
    question: str,
    df: pd.DataFrame,
    group_by: str = "",
    metric: str = "",
    date_grain: str = "",
    chart_type: str = "bar",
) -> dict[str, str]:
    selected_metric = pick_metric(question, df, metric)
    groups = infer_group_columns(question, df, f"{group_by} {date_grain}")
    select_parts = groups.copy()
    group_parts = ["month" if " AS month" in item else item for item in groups]

    aggregate_parts = [f"SUM({selected_metric}) AS {selected_metric}"]
    if selected_metric != "revenue" and "revenue" in df.columns:
        aggregate_parts.append("SUM(revenue) AS revenue")
    if selected_metric != "profit" and "profit" in df.columns:
        aggregate_parts.append("SUM(profit) AS profit")

    select_sql = ", ".join(select_parts + aggregate_parts)
    group_sql = ", ".join(group_parts)
    order_metric = selected_metric if selected_metric in df.columns else aggregate_parts[0].split(" AS ")[-1]
    where_sql = infer_where_clause(question, df)

    if group_sql:
        sql = f"SELECT {select_sql} FROM {TABLE_NAME} {where_sql} GROUP BY {group_sql} ORDER BY {group_sql}"
    else:
        sql = f"SELECT {', '.join(aggregate_parts)} FROM {TABLE_NAME} {where_sql}"

    x = "month" if any(" AS month" in group for group in groups) else (group_parts[0] if group_parts else order_metric)
    color_candidates = [group for group in group_parts if group != x]
    inferred_chart = infer_chart_type(question, chart_type)
    if any(" AS month" in group for group in groups) and inferred_chart != "pie":
        inferred_chart = "line"

    return {
        "intent": f"Analyze {selected_metric} for: {question}",
        "sql": sql,
        "chart_type": inferred_chart,
        "chart_x": x,
        "chart_y": selected_metric,
        "chart_color": color_candidates[0] if color_candidates else "",
    }
