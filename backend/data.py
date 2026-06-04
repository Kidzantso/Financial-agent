from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

from backend.config import TABLE_NAME


@dataclass
class AnalysisSummary:
    total_revenue: float
    total_profit: float
    profit_margin: float
    rows_returned: int
    top_dimension: str
    weak_dimension: str


def clean_column_name(name: str) -> str:
    """
    Standardizes a column name by lowercasing it and replacing non-alphanumeric 
    characters with underscores. Returns 'column' if the resulting string is empty.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "column"


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans all column names and attempts to parse columns containing the word 'date' 
    into proper datetime objects.
    """
    normalized = df.copy()
    normalized.columns = [clean_column_name(col) for col in normalized.columns]
    for col in normalized.columns:
        if "date" in col:
            parsed = pd.to_datetime(normalized[col], errors="coerce")
            if parsed.notna().any():
                normalized[col] = parsed
    return normalized


def dataframe_schema(df: pd.DataFrame) -> str:
    """
    Generates a string representation of the dataframe's schema (columns and types) 
    along with a 5-row sample. Used to provide context to the LLM.
    """
    lines = [f"Table name: {TABLE_NAME}", "Columns:"]
    for col, dtype in df.dtypes.items():
        lines.append(f"- {col}: {dtype}")
    lines.append("Sample rows:")
    lines.append(df.head(5).to_csv(index=False))
    return "\n".join(lines)


def sanitize_sql(sql: str) -> str:
    """
    Cleans and validates the generated SQL query. Ensures the query is a SELECT statement 
    and does not contain malicious operations like INSERT or DROP.
    """
    cleaned = sql.strip().strip("`")
    cleaned = re.sub(r"^sql\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if not re.match(r"^\s*select\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    blocked = r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma)\b"
    if re.search(blocked, cleaned, flags=re.IGNORECASE):
        raise ValueError("Unsafe SQL keyword detected.")
    return cleaned.rstrip(";")


def execute_sql_on_dataframe(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """
    Copies the dataframe into an in-memory SQLite database and executes the given 
    SQL query, returning the result as a new dataframe.
    """
    conn = sqlite3.connect(":memory:")
    try:
        sql_df = df.copy()
        for col in sql_df.select_dtypes(include=["datetime64[ns]"]).columns:
            sql_df[col] = sql_df[col].dt.strftime("%Y-%m-%d")
        sql_df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def summarize_result(result_df: pd.DataFrame, source_df: pd.DataFrame | None = None) -> AnalysisSummary:
    """
    Calculates key high-level metrics (total revenue, profit, margin, etc.) from the 
    resulting dataframe after a query has been executed.
    """
    if result_df.empty and source_df is None:
        return AnalysisSummary(0.0, 0.0, 0.0, 0, "n/a", "n/a")

    numeric_cols = list(result_df.select_dtypes("number").columns)
    revenue_col = "revenue" if "revenue" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
    profit_col = "profit" if "profit" in numeric_cols else None
    dimension_cols = [c for c in result_df.columns if c not in numeric_cols]
    dimension = dimension_cols[-1] if dimension_cols else (result_df.columns[0] if len(result_df.columns) else "")

    total_revenue = float(result_df[revenue_col].sum()) if revenue_col else 0.0
    total_profit = float(result_df[profit_col].sum()) if profit_col else 0.0

    if source_df is not None:
        source_revenue = (
            float(pd.to_numeric(source_df["revenue"], errors="coerce").sum())
            if "revenue" in source_df.columns
            else 0.0
        )
        source_profit = (
            float(pd.to_numeric(source_df["profit"], errors="coerce").sum())
            if "profit" in source_df.columns
            else 0.0
        )
        if total_revenue == 0.0 and source_revenue:
            total_revenue = source_revenue
        if (profit_col is None or total_profit == 0.0) and source_profit:
            total_profit = source_profit

    margin = (total_profit / total_revenue) if total_revenue else 0.0

    top_dimension = "n/a"
    weak_dimension = "n/a"
    if revenue_col and dimension in result_df:
        grouped = result_df.groupby(dimension, dropna=False)[revenue_col].sum().sort_values()
        if not grouped.empty:
            weak_dimension = str(grouped.index[0])
            top_dimension = str(grouped.index[-1])

    return AnalysisSummary(
        total_revenue=total_revenue,
        total_profit=total_profit,
        profit_margin=margin,
        rows_returned=len(result_df),
        top_dimension=top_dimension,
        weak_dimension=weak_dimension,
    )
