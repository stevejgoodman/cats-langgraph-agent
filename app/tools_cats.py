"""Toolbelt assembly for the cats-aware agent.

Extends the base tool belt with cat breed MCP tools (get_random_cats,
search_cat, get_cats_by_origin) from rag_with_cats, replacing the
feline-only RAG tool with one that also handles cat breed queries.
"""
from __future__ import annotations

from typing import List

from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from app.rag_with_cats import retrieve_information, get_random_cats, search_cat, get_cats_by_origin


def get_tool_belt() -> List:
    """Return tools: Tavily, Arxiv, RAG, and the three Cats MCP tools."""
    return [
        TavilySearch(max_results=5),
        ArxivQueryRun(),
        retrieve_information,
        get_random_cats,
        search_cat,
        get_cats_by_origin,
    ]
