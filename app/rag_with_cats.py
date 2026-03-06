"""Retrieval-Augmented Generation (RAG) utilities and tool, plus Cats MCP tools.

This module builds an in-memory RAG pipeline that:
- Loads PDF documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using a token-aware splitter.
- Embeds chunks with OpenAI and stores vectors in an in-memory Qdrant store.
- Exposes a LangChain Tool `retrieve_information` that retrieves relevant
  context and generates a response constrained to that context.

Additionally exposes `get_cats_mcp_tools()` which dynamically discovers tools
from the Cats MCP server using langchain-mcp-adapters. Tools are cached after
first load. Authentication uses a GCP service account identity token.
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Annotated, TypedDict

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, StateGraph

# ---------------------------------------------------------------------------
# RAG pipeline (unchanged from rag.py)
# ---------------------------------------------------------------------------



def _tiktoken_len(text: str) -> int:
    """Return token length using tiktoken; used for chunk length measurement."""
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: list[Document]
    response: str


def _build_rag_graph(data_dir: str):
    """Construct and compile a minimal RAG graph."""
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_vectorstore = QdrantVectorStore.from_documents(
        documents=chunks, embedding=embedding_model, location=":memory:", collection_name="rag_collection"
    )
    retriever = qdrant_vectorstore.as_retriever()

    human_template = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provide context to answer the provided user query. "
        "Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with \"I don't know\""
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", human_template)])
    generator_llm = ChatOpenAI(model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-nano"))

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"query": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
):
    """Use Retrieval Augmented Generation to retrieve information about feline health, including life stage care, nutrition, vaccinations, parasite control, behavior, diagnostics, and veterinary guidelines for cats."""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result


# ---------------------------------------------------------------------------
# Cat MCP tools — dynamically discovered via langchain-mcp-adapters
# ---------------------------------------------------------------------------

from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    from .python_client_iam_mcp import IAMAuthenticatedMCPClient
except ImportError:
    from python_client_iam_mcp import IAMAuthenticatedMCPClient  # type: ignore[no-redef]

_CATS_BASE_URL = "https://cats-mcp-660196542212.us-central1.run.app"
_CATS_MCP_URL = f"{_CATS_BASE_URL}/mcp"

# Resolve service account credentials relative to the project root (local dev)
_credentials_path = Path(__file__).parent.parent / ".secrets" / "fionaa-service-acct.json"
if _credentials_path.exists():
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(_credentials_path.resolve()))

_cats_auth_client = IAMAuthenticatedMCPClient(_CATS_BASE_URL)

# Module-level client kept alive so tools retain auth headers across invocations.
# Using async with would close the client after get_tools(), losing the config.
_cats_mcp_client: MultiServerMCPClient | None = None


def _get_cats_mcp_client() -> MultiServerMCPClient:
    global _cats_mcp_client
    if _cats_mcp_client is None:
        token = _cats_auth_client._get_identity_token()
        _cats_mcp_client = MultiServerMCPClient({
            "cats": {
                "transport": "streamable_http",
                "url": _CATS_MCP_URL,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        })
    return _cats_mcp_client


@lru_cache(maxsize=1)
def get_cats_mcp_tools() -> list:
    """Return MCP-discovered cat tools, loaded once and cached."""
    async def _load():
        return await _get_cats_mcp_client().get_tools()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_load())
    finally:
        loop.close()
