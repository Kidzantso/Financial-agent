# AI-Powered Business Intelligence Analyst

Streamlit app that turns natural-language business questions into tool-driven business analysis, charts, and an executive summary using Groq.

## Structure

```text
backend/
  agent.py      LangGraph + Groq tool-calling agent
  tools.py      Dataset tools exposed to the LLM
  analysis.py   Business analysis planning
  data.py       CSV normalization, SQL execution, KPI summaries
  config.py     Shared constants and prompts
frontend/
  app.py        Streamlit UI
app.py          Streamlit launcher
```

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Paste your Groq API key in the sidebar,upload a CSV, then ask a question such as:

```text
Show me revenue trends by region for Q4 2025 and highlight underperforming products.
```

A Groq API key is required.
