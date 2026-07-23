import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import memory

MCP_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_HISTORY_TURNS = 6

BASE_SYSTEM_PROMPT = (
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
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def _run_turn(history: list, user_input: str, system_prompt: str, text_box, trace_box) -> AIMessage:
    """Opens a fresh MCP connection, runs one turn end to end, streaming tool
    calls/results into trace_box and the reply into text_box. Returns the final
    AIMessage so the caller can save clean (Human, AI) pairs to session history."""
    model = ChatOpenAI(model=OPENAI_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    async with streamable_http_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            app = _build_graph(model, tools, system_prompt)

            messages = history + [HumanMessage(content=user_input)]
            streamed_text = ""
            final_ai_message = None
            async for mode, chunk in app.astream({"messages": messages}, stream_mode=["messages", "updates"]):
                if mode == "updates":
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
    return final_ai_message


st.set_page_config(page_title="MCP Assistant", page_icon="🔧")
st.title("MCP Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    language = st.text_input("Reply language", value="", placeholder="auto (e.g. Hindi, Spanish)")
    st.caption(f"Model: {OPENAI_MODEL}")
    st.caption(f"MCP server: {MCP_URL}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else "assistant"
    st.chat_message(role).markdown(_content_text(m.content))

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
            final_ai_message = asyncio.run(
                _run_turn(st.session_state.messages, user_input, system_prompt, text_box, trace_box)
            )
        except Exception as e:
            text_box.markdown(f"(model/tool call failed: {e})")
            final_ai_message = None

    if final_ai_message is not None:
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(final_ai_message)
        memory.remember_turn(user_input, _content_text(final_ai_message.content))
