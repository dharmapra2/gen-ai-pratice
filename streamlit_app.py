import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import Command
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import memory
from graph_utils import make_tools_node

MCP_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_HISTORY_TURNS = 6

BASE_SYSTEM_PROMPT = (
    "You are an assistant with tools for Slack, GitHub, Notion, Google Workspace "
    "(Gmail/Calendar/Drive), a read-only Postgres database, and a Chroma vector "
    "database (chroma_query for semantic search over a collection, chroma_add_documents "
    "to store new text). Translate the user's "
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
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= max_turns:
        return messages
    return messages[human_idxs[-max_turns]:]


def _build_graph(model, tools, system_prompt: str):
    model_with_tools = model.bind_tools(tools)

    async def agent_node(state: MessagesState):
        trimmed = _trim_for_model(state["messages"], MAX_HISTORY_TURNS)
        response = await model_with_tools.ainvoke([SystemMessage(content=system_prompt), *trimmed])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", make_tools_node(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    # Same checkpointer instance across reruns (session_state), keyed by a thread_id
    # that's stable for the browser session — required so a write-tool confirmation
    # gate can pause here and resume on a *later* rerun, after a fresh MCP connection
    # and a brand new compiled graph object (LangGraph resumes from checkpointed state,
    # not from Python object identity).
    return graph.compile(checkpointer=st.session_state.checkpointer)


async def _run_turn(graph_input, system_prompt: str, text_box, trace_box):
    """Opens a fresh MCP connection and streams one graph run: tool calls/results
    live into trace_box, the reply token by token into text_box. Returns
    ("done", AIMessage) on a normal finish, or ("pending", {"tool":.., "args":..})
    if the graph paused on a write tool's confirmation gate."""
    model = ChatOpenAI(model=OPENAI_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            app = _build_graph(model, tools, system_prompt)

            streamed_text = ""
            final_ai_message = None
            async for mode, chunk in app.astream(graph_input, config=config, stream_mode=["messages", "updates"]):
                if mode == "updates":
                    if "__interrupt__" in chunk:
                        return "pending", chunk["__interrupt__"][0].value
                    for node_name, update in chunk.items():
                        for m in update.get("messages", []):
                            if node_name == "agent":
                                for tc in m.tool_calls or []:
                                    trace_box.markdown(f"🔧 **{tc['name']}**`{tc['args']}`")
                                if not m.tool_calls:
                                    final_ai_message = m
                            elif node_name == "tools":
                                with trace_box.expander(f"↩ {m.name} result"):
                                    st.code(_content_text(m.content))
                elif mode == "messages":
                    msg_chunk, metadata = chunk
                    if metadata.get("langgraph_node") != "agent":
                        continue
                    text = _content_text(msg_chunk.content)
                    if text:
                        streamed_text += text
                        text_box.markdown(streamed_text)

    if final_ai_message is None:
        final_ai_message = AIMessage(content=streamed_text or "(empty response from model, try rephrasing)")
    return "done", final_ai_message


def _finish_turn(kind: str, result, user_input: str, system_prompt: str) -> None:
    """Handles the ("done", AIMessage) / ("pending", {...}) result of a turn."""
    if kind == "pending":
        st.session_state.pending = {**result, "user_input": user_input, "system_prompt": system_prompt}
    else:
        st.session_state.pending = None
        if result is not None:
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(result)
            memory.remember_turn(user_input, _content_text(result.content))
    st.rerun()


st.set_page_config(page_title="MCP Assistant", page_icon="🔧")
st.title("MCP Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
    st.session_state.thread_id = str(uuid.uuid4())
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    language = st.text_input("Reply language", value="", placeholder="auto (e.g. Hindi, Spanish)")
    st.caption(f"Model: {OPENAI_MODEL}")
    st.caption(f"MCP server: {MCP_URL}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pending = None
        st.rerun()

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    st.chat_message(role).markdown(_content_text(m.content))

if st.session_state.pending is not None:
    p = st.session_state.pending
    with st.chat_message("assistant"):
        st.warning(f"Confirm running **{p['tool']}**`{p['args']}`?")
        col1, col2 = st.columns(2)
        confirm = col1.button("Confirm", use_container_width=True)
        cancel = col2.button("Cancel", use_container_width=True)
    if confirm or cancel:
        with st.chat_message("assistant"):
            trace_box = st.container()
            text_box = st.empty()
            try:
                kind, result = asyncio.run(
                    _run_turn(Command(resume=confirm), p["system_prompt"], text_box, trace_box)
                )
            except Exception as e:
                text_box.markdown(f"(model/tool call failed: {e})")
                kind, result = "done", None
        _finish_turn(kind, result, p["user_input"], p["system_prompt"])
else:
    user_input = st.chat_input("Ask about Slack, GitHub, Notion, Gmail, Calendar, Drive, or the database...")
    if user_input:
        st.chat_message("user").markdown(user_input)

        system_prompt = BASE_SYSTEM_PROMPT
        if language.strip():
            system_prompt += (
                f"\n\nAlways reply in {language.strip()}, regardless of what language the "
                "user writes in or what language tool results come back in."
            )
        memories = memory.recall(user_input)
        if memories:
            system_prompt += f"\n\n{memories}"

        with st.chat_message("assistant"):
            trace_box = st.container()
            text_box = st.empty()
            try:
                kind, result = asyncio.run(
                    _run_turn({"messages": [HumanMessage(content=user_input)]}, system_prompt, text_box, trace_box)
                )
            except Exception as e:
                text_box.markdown(f"(model/tool call failed: {e})")
                kind, result = "done", None
        _finish_turn(kind, result, user_input, system_prompt)
