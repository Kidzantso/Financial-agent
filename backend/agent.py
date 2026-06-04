from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from backend.config import TOOL_AGENT_PROMPT
from backend.data import summarize_result
from backend.tools import get_tools


def should_continue_tool_agent(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def run_bi_agent(question: str, api_key: str, model: str, df: pd.DataFrame) -> dict[str, Any]:
    if not api_key.strip():
        raise ValueError("Groq API key is required.")

    tools = get_tools(df)
    llm = ChatGroq(groq_api_key=api_key, model=model, temperature=0).bind_tools(tools)

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue_tool_agent)
    workflow.add_edge("tools", "agent")
    graph = workflow.compile()

    response = graph.invoke(
        {
            "messages": [
                SystemMessage(content=TOOL_AGENT_PROMPT),
                HumanMessage(content=question),
            ]
        }
    )

    tool_payload: dict[str, Any] | None = None
    tool_trace: list[str] = []
    final_report = ""

    for message in response["messages"]:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            tool_trace.append(f"Model requested tool: {call.get('name')}")
        if getattr(message, "type", "") == "tool":
            tool_trace.append(f"Tool executed: {message.name}")
            if message.name == "analyze_business_data":
                tool_payload = json.loads(message.content)
        if getattr(message, "type", "") == "ai" and not tool_calls and message.content:
            final_report = str(message.content)

    if not tool_payload:
        raise ValueError("The model did not call analyze_business_data. Try again or use a stronger Groq model.")

    result_df = pd.DataFrame(tool_payload.get("rows", []))
    summary = summarize_result(result_df, df)
    analysis = {
        "rows_returned": summary.rows_returned,
        "total_revenue": summary.total_revenue,
        "total_profit": summary.total_profit,
        "profit_margin": summary.profit_margin,
        "top_dimension": summary.top_dimension,
        "weak_dimension": summary.weak_dimension,
    }

    return {
        "plan": {
            "intent": tool_payload.get("intent", "Business intelligence analysis"),
            "sql": tool_payload["sql"],
            "chart_type": tool_payload.get("chart_type", "bar"),
            "chart_x": tool_payload.get("chart_x", ""),
            "chart_y": tool_payload.get("chart_y", ""),
            "chart_color": tool_payload.get("chart_color", ""),
        },
        "sql": tool_payload["sql"],
        "result_df": result_df,
        "analysis": analysis,
        "report": final_report,
        "tool_trace": tool_trace,
    }
