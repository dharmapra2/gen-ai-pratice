import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_groq import ChatGroq
from langchain.agents import create_agent

MCP_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = (
    "You are an assistant with tools for Slack, GitHub, Notion, Google Workspace "
    "(Gmail/Calendar/Drive), and a read-only Postgres database. Translate the user's "
    "natural language request into correct tool arguments (e.g. Gmail search syntax "
    "for gmail_list_messages) and use the tools to answer. For 'has anyone discussed X' "
    "or 'find messages about X' style Slack questions, use slack_search_messages "
    "directly with the topic as the query — do not list channels and read each one's "
    "history one by one, and never guess a channel name/ID."
)


def _content_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content or ""


def _is_human(m) -> bool:
    if isinstance(m, dict):
        return m.get("role") == "user"
    return type(m).__name__ == "HumanMessage"


def _trim_history(messages: list, max_turns: int) -> list:
    """Keep only the last max_turns user turns (and everything after the first of
    them), so token usage per call stays roughly flat instead of growing with the
    whole conversation. Trims on turn boundaries so tool_call/tool_result pairs are
    never split apart."""
    human_idxs = [i for i, m in enumerate(messages) if _is_human(m)]
    if len(human_idxs) <= max_turns:
        return messages
    return messages[human_idxs[-max_turns]:]


def _print_trace(new_messages) -> None:
    for m in new_messages:
        for tc in getattr(m, "tool_calls", None) or []:
            print(f"  [tool call] {tc['name']}({tc['args']})")
        if type(m).__name__ == "ToolMessage":
            print(f"  [tool result] {m.name}: {_content_text(m.content)}")


async def _run_turn(agent, messages: list, prev_len: int) -> str:
    """Invoke the agent, retrying once on an empty completion or a transient API error
    (both happen occasionally on free-tier providers, e.g. Groq's tool_use_failed)."""
    for attempt in range(2):
        try:
            result = await agent.ainvoke({"messages": messages})
        except Exception as e:
            if attempt == 1:
                return messages, f"(model/tool call failed: {e})"
            continue
        _print_trace(result["messages"][prev_len:])
        reply = _content_text(result["messages"][-1].content)
        if reply or attempt == 1:
            return result["messages"], reply or "(empty response from model, try rephrasing)"
    return messages, "(no response)"


async def main():
    model = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)

            print(f"Connected. {len(tools)} tools available. Type 'exit' to quit.\n")
            messages = []
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue
                messages.append({"role": "user", "content": user_input})
                messages = _trim_history(messages, MAX_HISTORY_TURNS)
                prev_len = len(messages)
                messages, reply = await _run_turn(agent, messages, prev_len)
                print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
