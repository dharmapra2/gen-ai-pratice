import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_HISTORY_TURNS = 6
THREAD_ID = "cli-session"

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


def _trim_for_model(messages: list, max_turns: int) -> list:
    """Only the last max_turns user turns get sent to the model, so token usage per
    call stays roughly flat instead of growing with the whole conversation. The full
    history still lives in the checkpointer either way."""
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= max_turns:
        return messages
    return messages[human_idxs[-max_turns]:]


def build_graph(model, tools):
    model_with_tools = model.bind_tools(tools)

    async def agent_node(state: MessagesState):
        trimmed = _trim_for_model(state["messages"], MAX_HISTORY_TURNS)
        response = await model_with_tools.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *trimmed])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


async def _stream_once(app, config, input_messages: list) -> bool:
    """Streams one graph run: tool calls/results print live as nodes complete, and
    the assistant's reply streams token by token. Returns whether any text was shown."""
    printed_any = False
    printed_prefix = False
    async for mode, chunk in app.astream(
        {"messages": input_messages}, config=config, stream_mode=["messages", "updates"]
    ):
        if mode == "updates":
            for node_name, update in chunk.items():
                for m in update.get("messages", []):
                    if node_name == "agent":
                        for tc in getattr(m, "tool_calls", None) or []:
                            print(f"\n  [tool call] {tc['name']}({tc['args']})")
                    elif node_name == "tools":
                        print(f"\n  [tool result] {m.name}: {_content_text(m.content)}")
        elif mode == "messages":
            msg_chunk, metadata = chunk
            if metadata.get("langgraph_node") != "agent":
                continue
            text = _content_text(msg_chunk.content)
            if text:
                if not printed_prefix:
                    print("\nAssistant: ", end="", flush=True)
                    printed_prefix = True
                print(text, end="", flush=True)
                printed_any = True
    return printed_any


async def chat_turn(app, config, user_input: str) -> None:
    """Retries once on an empty completion or a transient API error. On retry the
    user message isn't resent — it's already in the checkpointer from the first
    attempt, so we just re-run the graph from that persisted state."""
    input_messages = [HumanMessage(content=user_input)]
    for attempt in range(2):
        try:
            printed_any = await _stream_once(app, config, input_messages)
        except Exception as e:
            if attempt == 0:
                input_messages = []
                continue
            print(f"\nAssistant: (model/tool call failed: {e})\n")
            return
        if printed_any:
            print("\n")
            return
        if attempt == 0:
            input_messages = []
            continue
        print("\nAssistant: (empty response from model, try rephrasing)\n")


async def main():
    model = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            app = build_graph(model, tools)
            config = {"configurable": {"thread_id": THREAD_ID}}

            print(f"Connected. {len(tools)} tools available. Type 'exit' to quit.\n")
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue
                await chat_turn(app, config, user_input)


if __name__ == "__main__":
    asyncio.run(main())
