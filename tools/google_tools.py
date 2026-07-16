import base64
import os
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .mcp_instance import mcp

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
]

_CREDENTIALS_DIR = Path(os.environ.get("GOOGLE_CREDENTIALS_DIR", "credentials"))
_CLIENT_SECRET_FILE = _CREDENTIALS_DIR / "client_secret.json"
_TOKEN_FILE = _CREDENTIALS_DIR / "token.json"


def _get_creds() -> Credentials:
    creds = None
    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(_CLIENT_SECRET_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        _TOKEN_FILE.write_text(creds.to_json())
    return creds


def _gmail():
    return build("gmail", "v1", credentials=_get_creds())


def _calendar():
    return build("calendar", "v3", credentials=_get_creds())


def _drive():
    return build("drive", "v3", credentials=_get_creds())


@mcp.tool()
def gmail_list_messages(query: str = "", max_results: int = 10) -> list[dict]:
    """List Gmail messages matching a search query (Gmail search syntax, e.g. 'from:x@y.com is:unread')"""
    service = _gmail()
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    messages = []
    for m in resp.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=m["id"], format="metadata", metadataHeaders=["Subject", "From"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        messages.append(
            {
                "id": msg["id"],
                "subject": headers.get("Subject", ""),
                "from": headers.get("From", ""),
                "snippet": msg.get("snippet", ""),
            }
        )
    return messages


@mcp.tool()
def gmail_send_message(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail. Returns the sent message ID."""
    service = _gmail()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return sent["id"]


@mcp.tool()
def calendar_list_events(calendar_id: str = "primary", max_results: int = 10) -> list[dict]:
    """List upcoming events on a Google Calendar"""
    service = _calendar()
    now = datetime.now(timezone.utc).isoformat()
    resp = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return [
        {
            "id": e["id"],
            "summary": e.get("summary", ""),
            "start": e["start"].get("dateTime", e["start"].get("date")),
            "end": e["end"].get("dateTime", e["end"].get("date")),
        }
        for e in resp.get("items", [])
    ]


@mcp.tool()
def calendar_create_event(summary: str, start: str, end: str, calendar_id: str = "primary") -> str:
    """Create a Google Calendar event. start/end are ISO 8601 datetimes. Returns the event link."""
    service = _calendar()
    event = {"summary": summary, "start": {"dateTime": start}, "end": {"dateTime": end}}
    created = service.events().insert(calendarId=calendar_id, body=event).execute()
    return created.get("htmlLink")


@mcp.tool()
def drive_list_files(query: str = "", max_results: int = 10) -> list[dict]:
    """List Google Drive files, optionally filtered by a Drive query string"""
    service = _drive()
    resp = (
        service.files()
        .list(q=query or None, pageSize=max_results, fields="files(id, name, mimeType)")
        .execute()
    )
    return resp.get("files", [])


@mcp.tool()
def drive_read_file(file_id: str) -> str:
    """Read the text content of a Google Drive file (plain text or exported Google Doc)"""
    service = _drive()
    meta = service.files().get(fileId=file_id, fields="mimeType").execute()
    if meta["mimeType"] == "application/vnd.google-apps.document":
        data = service.files().export(fileId=file_id, mimeType="text/plain").execute()
    else:
        data = service.files().get_media(fileId=file_id).execute()
    return data.decode("utf-8") if isinstance(data, bytes) else data
