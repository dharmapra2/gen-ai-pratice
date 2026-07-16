from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sample-mcp")
