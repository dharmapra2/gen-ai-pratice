import os
import uuid

import chromadb

from .mcp_instance import mcp


def _client() -> chromadb.ClientAPI:
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ.get("CHROMA_TENANT"),
        database=os.environ.get("CHROMA_DATABASE"),
    )


@mcp.tool()
def chroma_list_collections() -> list[str]:
    """List Chroma collection names"""
    return [c.name for c in _client().list_collections()]


@mcp.tool()
def chroma_create_collection(name: str) -> str:
    """Create a Chroma collection if it doesn't already exist. Returns the collection name."""
    _client().get_or_create_collection(name=name)
    return name


@mcp.tool()
def chroma_delete_collection(name: str) -> str:
    """Delete a Chroma collection"""
    _client().delete_collection(name=name)
    return f"deleted {name}"


@mcp.tool()
def chroma_add_documents(collection: str, documents: list[str], metadatas: list[dict] = None) -> list[str]:
    """Add text documents to a Chroma collection (embeddings generated automatically).
    Returns the generated document IDs."""
    coll = _client().get_or_create_collection(name=collection)
    ids = [uuid.uuid4().hex for _ in documents]
    coll.add(documents=documents, ids=ids, metadatas=metadatas)
    return ids


@mcp.tool()
def chroma_query(collection: str, query_text: str, n_results: int = 5) -> list[dict]:
    """Query a Chroma collection for documents most similar to query_text"""
    coll = _client().get_collection(name=collection)
    resp = coll.query(query_texts=[query_text], n_results=n_results)
    return [
        {"id": doc_id, "document": doc, "metadata": meta, "distance": dist}
        for doc_id, doc, meta, dist in zip(
            resp["ids"][0], resp["documents"][0], resp["metadatas"][0], resp["distances"][0]
        )
    ]
