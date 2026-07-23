# gen-ai-practice — MCP server + LangGraph agent

An MCP (Model Context Protocol) server exposing tools for Slack, GitHub, Notion,
Google Workspace (Gmail/Calendar/Drive), and a read-only Postgres database — plus a
terminal chat client that connects to it through a LangGraph agent (OpenAI-backed).

## Architecture

```
server.py            MCP server (streamable-http, port 8000)
  tools/
    mcp_instance.py    shared FastMCP instance + .env loading
    slack_tools.py      list/read/search channels
    github_tools.py     repos, issues, PRs, file contents
    notion_tools.py     search, read, query, create pages
    google_tools.py     Gmail, Calendar, Drive (OAuth installed-app flow)
    db_tools.py          read-only Postgres queries

agent.py              Terminal chat client
  - connects to server.py over MCP (streamable-http)
  - loads all MCP tools via langchain-mcp-adapters
  - explicit LangGraph StateGraph: agent node (OpenAI + tools) <-> tools node
  - InMemorySaver checkpointer for in-process conversation memory
  - streams tokens + tool calls live to the terminal

streamlit_app.py      Web chat client (same graph shape as agent.py)
  - opens a fresh MCP connection per turn instead of one long-lived one
    (Streamlit reruns the whole script on every interaction)
  - keeps conversation history in st.session_state instead of a checkpointer
  - same streaming tool-call/reply display, as chat bubbles

memory.py             Mem0 wrapper — used by both clients above
  - recall(query): fetch facts relevant to the current message
  - remember_turn(user_input, reply): store the turn for future sessions
  - persists across process restarts (unlike the two clients' own history)
```

The two processes are independent: `server.py` just serves MCP tools over HTTP;
`agent.py` / `streamlit_app.py` (or any other MCP client — Claude Code, MCP
Inspector, etc.) is what turns natural language into tool calls.

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv        # if venv/ doesn't already exist
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Fill in `.env` — see the per-integration notes below for how to get each value.
Only fill in the integrations you actually plan to use; a tool will only fail when
you call it, not at server startup.

#### Slack

Uses a **user token** (`xoxp-...`), not a bot token — so it reads/searches whatever
channels/DMs you personally have access to, without needing to invite a bot to
every channel.

1. Create a Slack app at [api.slack.com/apps](https://api.slack.com/apps) (or reuse
   one) and add these **User Token Scopes**: `channels:read`, `channels:history`,
   `groups:history`, `users:read`, `search:read`.
2. Install the app to your workspace, copy the `xoxp-...` user token.
3. Set `SLACK_USER_TOKEN` in `.env`.

#### GitHub

1. Create a [personal access token](https://github.com/settings/tokens) with `repo`
   scope.
2. Set `GITHUB_TOKEN` in `.env`.

#### Notion

1. Create an [internal integration](https://www.notion.so/my-integrations) and copy
   its secret.
2. Share the specific pages/databases you want accessible with that integration
   (Notion won't expose anything not explicitly shared).
3. Set `NOTION_TOKEN` in `.env`.

#### Postgres

Set `DATABASE_URL` to a plain `postgresql://user:pass@host:port/dbname` URL — **not**
a SQLAlchemy-style URL like `postgresql+asyncpg://...`, `psycopg2` doesn't parse the
`+driver` suffix.

The DB tools are read-only by design: `db_query` only allows `SELECT`/`WITH`/`EXPLAIN`
(regex-checked *and* the connection itself is opened in a read-only Postgres session,
so even a data-modifying CTE like `WITH x AS (DELETE ...) SELECT * FROM x` is rejected
at the database level, not just by the regex).

#### Google Workspace (Gmail / Calendar / Drive)

Uses the OAuth **installed-app flow**, not a service account — service accounts can't
access a personal Google account's Gmail/Drive without domain-wide delegation (which
requires a Workspace admin), so this uses a one-time interactive browser login instead.

1. In [Google Cloud Console](https://console.cloud.google.com/), create an OAuth
   Client ID of type **Desktop app**.
2. Download the JSON and save it as `credentials/client_secret.json`.
3. The first time any Gmail/Calendar/Drive tool is called, a browser window opens to
   authorize. The resulting token is cached at `credentials/token.json` and reused
   (auto-refreshed) after that — no repeated logins.

`credentials/` and `*.json` are already in `.gitignore`, so these never get committed.

#### OpenAI (for `agent.py` / `streamlit_app.py`)

1. Get an API key from [platform.openai.com](https://platform.openai.com/api-keys).
2. Set `OPENAI_API_KEY` in `.env`. Defaults to `gpt-4.1-mini`; override with
   `OPENAI_MODEL` if you want a different model.

#### Mem0 (persistent memory)

Both clients use the hosted [Mem0](https://mem0.ai) platform to remember facts about
you across sessions — separate from each client's own in-process conversation
history, which resets when the process restarts.

1. Get a free API key from [app.mem0.ai](https://app.mem0.ai).
2. Set `MEM0_API_KEY` in `.env`.

If unset (or Mem0 is unreachable), both clients degrade gracefully — `memory.py`
catches the error and just returns no memories, it doesn't break the chat.

## Running it

**Terminal 1 — start the MCP server:**

```bash
python server.py
```

Serves over streamable-http at `http://127.0.0.1:8000/mcp`.

**Terminal 2 — start the chat agent:**

```bash
python agent.py
```

```
Connected. 23 tools available. Type 'exit' to quit.
Type '/lang <language>' to set the reply language (e.g. '/lang Hindi'), '/lang' to reset to auto.

You: has anyone discussed the introspeq project in slack?
  [tool call] slack_search_messages({'query': 'introspeq'})
  [tool result] slack_search_messages: ...

Assistant: Yes — several messages in #introspeq-ai and #introspeq-ai-internal...
```

Chat commands:
- `exit` / `quit` — end the session
- `/lang <language>` — switch reply language (e.g. `/lang Hindi`, `/lang Spanish`);
  `/lang` alone resets to auto/English. Applies to every subsequent turn, including
  tool-using ones.

Conversation memory persists for the life of the process (LangGraph checkpointer),
but only the last 6 user turns are actually sent to the model each call to keep
token usage roughly flat — full history stays in the checkpointer either way.
Durable facts (name, preferences, project details, etc.) persist across restarts
too, via Mem0 — see below.

**Terminal 2 (alternative) — or the web UI instead:**

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Same tools, same streaming tool-call/reply
behavior, chat bubbles instead of a terminal, and a "Reply language" field in the
sidebar instead of `/lang`.

## Testing tools without the agent

**MCP Inspector** (visual, manually call any tool with a form):

```bash
npx @modelcontextprotocol/inspector
```
Transport: Streamable HTTP, URL: `http://127.0.0.1:8000/mcp`.

**Register with Claude Code** (use tools through natural language in a Claude Code
session instead of `agent.py`):

```bash
claude mcp add --transport http mcp-tools http://127.0.0.1:8000/mcp
```

## Available tools

| Module | Tools |
|---|---|
| Slack | `slack_list_channels`, `slack_read_channel_history`, `slack_search_messages` |
| GitHub | `github_list_repos`, `github_list_issues`, `github_create_issue`, `github_list_pull_requests`, `github_get_file_contents` |
| Notion | `notion_search`, `notion_get_page`, `notion_query_database`, `notion_create_page` |
| Google | `gmail_list_messages`, `gmail_send_message`, `calendar_list_events`, `calendar_create_event`, `drive_list_files`, `drive_read_file` |
| Postgres | `db_query` (read-only), `db_list_tables`, `db_describe_table` |
| Sample | `add`, `greet` |

`slack_send_message` exists in `slack_tools.py` but is commented out (not registered)
— uncomment the `@mcp.tool()` line if you want the agent able to post messages, not
just read them.

## Notes / gotchas

- **Slack `not_in_channel` error**: with a bot token, the bot must be invited to a
  channel before reading its history. Using a user token (as configured here) avoids
  this since it reads on your own behalf.
- **Slack search results are truncated** to short previews (~300 chars) to avoid
  blowing up token usage / triggering request-size errors on large matches — use
  `slack_read_channel_history` near a match's `ts` to read one in full.
- **Free-tier LLM quotas are per-model and vary a lot per project** — if you swap
  `OPENAI_MODEL` for a free-tier provider's model, expect to hit daily/per-minute caps
  quickly on anything beyond light testing.
