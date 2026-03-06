"""Retrieval-Augmented Generation (RAG) utilities and tool, plus Cats MCP tools.

This module builds an in-memory RAG pipeline that:
- Loads PDF documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using a token-aware splitter.
- Embeds chunks with OpenAI and stores vectors in an in-memory Qdrant store.
- Exposes a LangChain Tool `retrieve_information` that retrieves relevant
  context and generates a response constrained to that context.

Additionally exposes three tools that call the Cats MCP server:
- get_random_cats: fetch N random cat breeds with origin/traits/lifespan info
- search_cat: look up a specific cat breed by name
- get_cats_by_origin: list cat breeds by country or region of origin

The MCP tools authenticate via a Google Cloud identity token obtained by
calling `gcloud auth print-identity-token` or via service account credentials.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
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
# Cat MCP tools
# ---------------------------------------------------------------------------

import json
import urllib.request
from pathlib import Path

try:
    from .python_client_iam_mcp import IAMAuthenticatedMCPClient
except ImportError:
    from python_client_iam_mcp import IAMAuthenticatedMCPClient  # type: ignore[no-redef]

_CATS_BASE_URL = "https://cats-mcp-660196542212.us-central1.run.app"
_CATS_MCP_URL = f"{_CATS_BASE_URL}/mcp"

# Resolve service account credentials relative to the project root
_credentials_path = Path(__file__).parent.parent / ".secrets" / "fionaa-service-acct.json"
if _credentials_path.exists():
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(_credentials_path.resolve()))

_cats_client = IAMAuthenticatedMCPClient(_CATS_BASE_URL)


def _call_cats_mcp(tool_name: str, arguments: dict) -> str:
    """Open a fresh MCP session, call one tool, and return the result string.

    Uses IAMAuthenticatedMCPClient for token acquisition, then manually drives
    the MCP session protocol which requires:
      1. An initialize request (returns mcp-session-id header)
      2. A tools/call request with that session ID and
         Accept: application/json, text/event-stream
    """
    token = _cats_client._get_identity_token()
    base_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def _post(payload: dict, session_id: str | None = None) -> tuple[str | None, dict]:
        headers = {**base_headers}
        if session_id:
            headers["mcp-session-id"] = session_id
        data = json.dumps(payload).encode()
        req = urllib.request.Request(_CATS_MCP_URL, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            sid = r.headers.get("mcp-session-id")
            body = r.read().decode()
        # SSE responses are prefixed with "event: message\ndata: ..."
        for line in body.splitlines():
            if line.startswith("data: "):
                body = line[len("data: "):]
                break
        return sid, json.loads(body)

    # 1. Initialize to get a session ID
    session_id, _ = _post({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "langgraph-cats-agent", "version": "1.0"},
        },
    })

    # 2. Call the tool within the session
    _, response = _post(
        {
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        session_id=session_id,
    )

    if "error" in response:
        raise RuntimeError(f"MCP error from {tool_name}: {response['error']}")

    content = response.get("result", {}).get("content", [])
    texts = [block["text"] for block in content if block.get("type") == "text"]
    return "\n".join(texts) if texts else json.dumps(response.get("result", {}))


@tool
def get_random_cats(
    count: Annotated[int, "Number of random cat breeds to fetch (1-20, default 5)"] = 5,
) -> str:
    """Fetch random cat breeds with information about their origin, weight, lifespan, and traits."""
    return _call_cats_mcp("get_random_cats", {"count": count})


@tool
def search_cat(
    name: Annotated[str, "Cat breed name to search for, e.g. 'siamese', 'persian', 'maine coon'"],
) -> str:
    """Search for a specific cat breed by name and return detailed information including origin, traits, and image."""
    return _call_cats_mcp("search_cat", {"name": name})


@tool
def get_cats_by_origin(
    origin: Annotated[str, "Country or region to filter by, e.g. 'egypt', 'united states', 'japan'"],
) -> str:
    """Get a list of cat breeds filtered by country or region of origin with their traits."""
    return _call_cats_mcp("get_cats_by_origin", {"origin": origin})
