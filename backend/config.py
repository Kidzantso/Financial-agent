import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = APP_DIR / "sample_sales_data.csv"
TABLE_NAME = "sales"
DEFAULT_QUESTION = os.getenv(
    "DEFAULT_QUESTION",
    "Show me revenue trends by region for Q4 2025 and highlight underperforming products."
)
GROQ_MODEL_OPTIONS = os.getenv("GROQ_MODEL_OPTIONS", "llama-3.1-8b-instant,llama-3.3-70b-versatile").split(",")

TOOL_AGENT_PROMPT = """You are an AI-powered business intelligence analyst.
Use tools only when the user's request requires dataset facts, calculations,
charts, comparisons, trends, or business recommendations.

For BI analysis requests:
1. Call inspect_dataset_schema if you need to understand the data.
2. Call analyze_business_data with the business question and optional grouping,
   metric, chart, or time-grain hints. Do not write SQL yourself.

After the tool returns results, write a concise executive summary with key
findings, underperformers, and recommended actions. Never claim analysis that
was not returned by the tool.
"""
