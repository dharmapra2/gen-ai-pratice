import os
import re

from slack_sdk import WebClient

from .mcp_instance import mcp

_MENTION_RE = re.compile(r"<@([A-Z0-9]+)>")
_user_name_cache: dict[str, str] = {}


def _client() -> WebClient:
    return WebClient(token=os.environ["SLACK_USER_TOKEN"])


def _resolve_user_name(client: WebClient, user_id: str) -> str:
    if user_id not in _user_name_cache:
        try:
            profile = client.users_info(user=user_id)["user"]
            _user_name_cache[user_id] = (
                profile.get("profile", {}).get("display_name")
                or profile.get("real_name")
                or user_id
            )
        except Exception:
            _user_name_cache[user_id] = user_id
    return _user_name_cache[user_id]


def _resolve_mentions(client: WebClient, text: str) -> str:
    return _MENTION_RE.sub(
        lambda m: f"@{_resolve_user_name(client, m.group(1))}", text
    )


def _truncate(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


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
    client = _client()
    resp = client.conversations_history(channel=channel, limit=limit)
    return [
        {
            "user": _resolve_user_name(client, m["user"]) if m.get("user") else None,
            "text": _truncate(_resolve_mentions(client, m.get("text", "")), 1000),
            "ts": m.get("ts"),
        }
        for m in resp["messages"]
    ]


@mcp.tool()
def slack_search_messages(query: str, count: int = 20) -> list[dict]:
    """Full-text search across all Slack channels/DMs the user is a member of.
    Use this for 'has anyone discussed X' or 'find messages about X' questions instead
    of listing channels and reading each one's history. Returns short previews — use
    slack_read_channel_history near a result's ts to read a match in full."""
    client = _client()
    resp = client.search_messages(query=query, count=count)
    matches = resp["messages"]["matches"]
    return [
        {
            "channel": m.get("channel", {}).get("name"),
            "user": _resolve_user_name(client, m["user"]) if m.get("user") else None,
            "text": _truncate(_resolve_mentions(client, m.get("text", ""))),
            "ts": m.get("ts"),
            "permalink": m.get("permalink"),
        }
        for m in matches
    ]
