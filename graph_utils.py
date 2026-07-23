from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

WRITE_TOOLS = {
    "github_create_issue",
    "notion_create_page",
    "gmail_send_message",
    "calendar_create_event",
    "chroma_create_collection",
    "chroma_delete_collection",
    "chroma_add_documents",
}


def make_tools_node(tools):
    """Builds a tools node that pauses on any WRITE_TOOLS call via a LangGraph
    interrupt — the tool physically cannot run until the caller resumes the graph
    with Command(resume=True/False). Read-only tools run immediately, no gate."""
    tools_by_name = {t.name: t for t in tools}

    async def tools_node(state):
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            name, args, call_id = call["name"], call["args"], call["id"]
            if name in WRITE_TOOLS and not interrupt({"tool": name, "args": args}):
                results.append(
                    ToolMessage(content=f"User declined to run {name}.", tool_call_id=call_id, name=name)
                )
                continue
            tool_call = {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            try:
                results.append(await tools_by_name[name].ainvoke(tool_call))
            except Exception as e:
                results.append(
                    ToolMessage(content=f"Error executing tool {name}: {e}", tool_call_id=call_id, name=name)
                )
        return {"messages": results}

    return tools_node
