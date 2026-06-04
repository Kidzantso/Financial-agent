from __future__ import annotations

import os
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.agent import run_bi_agent
from backend.config import DEFAULT_QUESTION, SAMPLE_DATA_PATH, GROQ_MODEL_OPTIONS
from backend.data import normalize_dataframe


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Loads and caches the default sample dataset."""
    return normalize_dataframe(pd.read_csv(SAMPLE_DATA_PATH))


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Reads an uploaded CSV file from Streamlit and normalizes it."""
    raw = uploaded_file.getvalue().decode("utf-8")
    return normalize_dataframe(pd.read_csv(StringIO(raw)))


def make_chart(result_df: pd.DataFrame, plan: dict):
    """
    Generates a Plotly chart based on the inferred chart type and axes.
    Falls back to a bar chart if no valid match is found.
    """
    chart_type = str(plan.get("chart_type", "bar")).lower()
    x = plan.get("chart_x")
    y = plan.get("chart_y")
    color = plan.get("chart_color")
    if x not in result_df.columns:
        x = result_df.columns[0]
    numeric_cols = list(result_df.select_dtypes("number").columns)
    if y not in result_df.columns:
        y = numeric_cols[0] if numeric_cols else result_df.columns[-1]
    if color not in result_df.columns:
        color = None

    if chart_type == "line":
        return px.line(result_df, x=x, y=y, color=color, markers=True)
    if chart_type == "scatter":
        return px.scatter(result_df, x=x, y=y, color=color, size=y if y in numeric_cols else None)
    if chart_type == "pie":
        return px.pie(result_df, names=x, values=y)
    return px.bar(result_df, x=x, y=y, color=color)


def render_metric(label: str, value: str) -> None:
    """Helper to render a Streamlit metric component."""
    st.metric(label, value)


def main() -> None:
    """Main entry point for the Streamlit frontend UI."""
    st.set_page_config(page_title="BI Analyst Agent", layout="wide")
    st.title("AI-Powered Business Intelligence Analyst")

    with st.sidebar:
        st.header("Model")
        api_key = st.text_input(
            "Groq API key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Required. Stored only in Streamlit session memory.",
        )
        model = st.selectbox(
            "Groq model",
            GROQ_MODEL_OPTIONS,
        )
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("No upload needed: demo Q4 2025 sales data is included.")

    df = load_uploaded_csv(uploaded_file) if uploaded_file else load_sample_data()
    question = st.text_area("Business question", value=DEFAULT_QUESTION, height=90)

    left, right = st.columns([0.7, 0.3])
    with left:
        run = st.button("Analyze", type="primary", use_container_width=True)
    with right:
        st.download_button(
            "Download current data",
            data=df.to_csv(index=False),
            file_name="bi_agent_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Data Preview", expanded=False):
        st.dataframe(df, use_container_width=True)

    if not run:
        return

    if not api_key.strip():
        st.error("Enter a Groq API key to run the agent.")
        return
    if not question.strip():
        st.warning("Enter a business question to analyze.")
        return

    with st.status("Running analyst agent...", expanded=True) as status:
        st.write("Sending the question to Groq")
        st.write("Allowing the model to call BI tools when needed")
        st.write("Rendering the returned analysis")
        try:
            final_state = run_bi_agent(
                question=question.strip(),
                api_key=api_key.strip(),
                model=model,
                df=df,
            )
            status.update(label="Analysis complete", state="complete")
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return

    analysis = final_state["analysis"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric("Revenue", f"${analysis['total_revenue']:,.0f}")
    with m2:
        render_metric("Profit", f"${analysis['total_profit']:,.0f}")
    with m3:
        render_metric("Margin", f"{analysis['profit_margin']:.1%}")
    with m4:
        render_metric("Rows", f"{analysis['rows_returned']:,}")

    st.subheader("Executive Summary")
    st.markdown(final_state["report"])

    if final_state.get("tool_trace"):
        with st.expander("Tool Calls", expanded=False):
            for item in final_state["tool_trace"]:
                st.write(item)

    chart = make_chart(final_state["result_df"], final_state["plan"])
    st.plotly_chart(chart, use_container_width=True)

    with st.expander("SQL and Results", expanded=True):
        st.code(final_state["sql"], language="sql")
        st.dataframe(final_state["result_df"], use_container_width=True)
