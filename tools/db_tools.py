import os
import re

import psycopg2
import psycopg2.extras

from .mcp_instance import mcp

_READ_ONLY_PATTERN = re.compile(r"^\s*(select|with|explain)\b", re.IGNORECASE)


def _connect():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.set_session(readonly=True, autocommit=True)
    return conn


@mcp.tool()
def db_query(sql: str) -> list[dict]:
    """Run a read-only SQL query (SELECT/WITH/EXPLAIN only) against the configured Postgres database"""
    if not _READ_ONLY_PATTERN.match(sql):
        raise ValueError("Only SELECT, WITH, and EXPLAIN queries are allowed")
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]


@mcp.tool()
def db_list_tables(schema: str = "public") -> list[str]:
    """List tables in a Postgres schema"""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
                (schema,),
            )
            return [row[0] for row in cur.fetchall()]


@mcp.tool()
def db_describe_table(table: str, schema: str = "public") -> list[dict]:
    """Describe the columns of a Postgres table"""
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (schema, table),
            )
            return [dict(row) for row in cur.fetchall()]
