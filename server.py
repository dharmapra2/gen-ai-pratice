from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sample-mcp")

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
