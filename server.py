from tools.mcp_instance import mcp
from tools import (  # noqa: F401
    slack_tools,
    github_tools,
    notion_tools,
    google_tools,
    db_tools,
    chroma_tools,
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Greet a user by name"""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
