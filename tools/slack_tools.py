import os

from slack_sdk import WebClient

from .mcp_instance import mcp


def _client() -> WebClient:
    return WebClient(token=os.environ["SLACK_USER_TOKEN"])


@mcp.tool()
def slack_list_channels(limit: int = 100) -> list[dict]:
    """List channels the bot can see in the Slack workspace"""
    resp = _client().conversations_list(limit=limit)
    return [{"id": c["id"], "name": c["name"]} for c in resp["channels"]]


# @mcp.tool()
# def slack_send_message(channel: str, text: str) -> str:
#     """Send a message to a Slack channel (name or ID). Returns the message timestamp."""
#     resp = _client().chat_postMessage(channel=channel, text=text)
#     return resp["ts"]


@mcp.tool()
def slack_read_channel_history(channel: str, limit: int = 20) -> list[dict]:
    """Read recent messages from a Slack channel (channel ID)"""
    resp = _client().conversations_history(channel=channel, limit=limit)
    return [
        {"user": m.get("user"), "text": m.get("text"), "ts": m.get("ts")}
        for m in resp["messages"]
    ]
