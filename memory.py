import os

from mem0 import MemoryClient

USER_ID = "local-user"

_client = None


def _get_client() -> MemoryClient:
    global _client
    if _client is None:
        _client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
    return _client


def recall(query: str) -> str:
    """Return remembered facts relevant to query as a system-prompt-ready block,
    or "" if there's nothing relevant (or Mem0 is unreachable/unconfigured)."""
    try:
        results = _get_client().search(query, filters={"user_id": USER_ID}, top_k=5)
    except Exception:
        return ""
    facts = [r.get("memory", "") for r in results.get("results", []) if r.get("memory")]
    if not facts:
        return ""
    return "Relevant things you remember about the user:\n" + "\n".join(f"- {f}" for f in facts)


def remember_turn(user_input: str, assistant_reply: str) -> None:
    """Store this turn so Mem0 can extract durable facts for future sessions."""
    if not assistant_reply:
        return
    try:
        _get_client().add(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_reply},
            ],
            user_id=USER_ID,
        )
    except Exception:
        pass
