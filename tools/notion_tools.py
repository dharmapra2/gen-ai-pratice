import os

from notion_client import Client

from .mcp_instance import mcp


def _client() -> Client:
    return Client(auth=os.environ["NOTION_TOKEN"])


def _extract_title(obj: dict) -> str:
    for prop in obj.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(p.get("plain_text", "") for p in prop.get("title", []))
    return obj.get("id", "")


@mcp.tool()
def notion_search(query: str) -> list[dict]:
    """Search Notion pages and databases by title"""
    resp = _client().search(query=query)
    return [
        {"id": r["id"], "type": r["object"], "title": _extract_title(r), "url": r.get("url")}
        for r in resp.get("results", [])
    ]


@mcp.tool()
def notion_get_page(page_id: str) -> dict:
    """Get a Notion page's properties"""
    return _client().pages.retrieve(page_id=page_id)


@mcp.tool()
def notion_query_database(database_id: str) -> list[dict]:
    """Query all pages in a Notion database"""
    resp = _client().databases.query(database_id=database_id)
    return resp.get("results", [])


@mcp.tool()
def notion_create_page(database_id: str, title: str) -> str:
    """Create a new page with a title in a Notion database. Returns the page URL."""
    client = _client()
    db = client.databases.retrieve(database_id=database_id)
    title_prop = next(name for name, prop in db["properties"].items() if prop["type"] == "title")
    page = client.pages.create(
        parent={"database_id": database_id},
        properties={title_prop: {"title": [{"text": {"content": title}}]}},
    )
    return page["url"]
