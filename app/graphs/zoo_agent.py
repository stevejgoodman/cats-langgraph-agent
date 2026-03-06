"""A tool-using agent graph with Zoo Animal MCP tools.

Identical flow to simple_agent but uses the extended tool belt from
tools_zoo, which adds get_random_animals, search_animal, and
get_animals_by_type alongside the existing Tavily, Arxiv, and RAG tools.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.models import get_chat_model
from app.state import MessagesState
from app.tools_cats import get_tool_belt


def _build_model_with_tools():
    return get_chat_model().bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    response = _build_model_with_tools().invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "action", END: END})
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
