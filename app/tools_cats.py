"""Toolbelt assembly for the cats-aware agent.

Extends the base tool belt with cat breed MCP tools dynamically discovered
from the Cats MCP server via langchain-mcp-adapters.
"""
from __future__ import annotations

from typing import List

from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag_with_cats import retrieve_information, get_cats_mcp_tools


def get_tool_belt() -> List:
    """Return tools: Tavily, Arxiv, RAG, and dynamically discovered Cats MCP tools."""
    return [
        TavilySearch(max_results=5),
        ArxivQueryRun(),
        retrieve_information,
        *get_cats_mcp_tools(),
    ]
