#!/usr/bin/env python3
"""
A web server where the AI makes it up as it goes along

This experimental server sends every HTTP request to an LLM with three tools:
- database: Execute SQL queries on SQLite
- webResponse: Generate HTTP responses (HTML, JSON, etc.)
- updateMemory: Save user feedback and preferences

Run with: python server.py
"""

import json
import os
import random
import sqlite3
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.json import JSON

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Application configuration loaded from environment variables."""

    port = int(os.getenv("PORT", 3001))
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    class Anthropic:
        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        api_key = os.getenv("ANTHROPIC_API_KEY")

    class OpenAI:
        model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        api_key = os.getenv("OPENAI_API_KEY")

    anthropic = Anthropic()
    openai = OpenAI()


config = Config()


# ============================================================================
# Logger
# ============================================================================

# Create rich console for beautiful logging
console = Console()


class Logger:
    """Structured logger with rich color-coded output."""

    @staticmethod
    def _log(area: str, message: str, data=None, color: str = "white", prefix: str = ""):
        """Internal logging helper."""
        timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        area_tag = f"[{area.upper()}]".ljust(12)
        msg = f"{prefix} {message}" if prefix else message
        console.print(f"[dim]{timestamp}[/dim] [{color}]{area_tag}[/{color}] [{color}]{msg}[/{color}]")
        if data is not None:
            console.print(JSON(json.dumps(data), indent=2), style="dim")

    @staticmethod
    def info(area: str, message: str, data=None):
        Logger._log(area, message, data, "cyan")

    @staticmethod
    def success(area: str, message: str, data=None):
        Logger._log(area, message, data, "green", "✅")

    @staticmethod
    def error(area: str, message: str, error=None):
        timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        area_tag = f"[{area.upper()}]".ljust(12)
        console.print(f"[dim]{timestamp}[/dim] [red]{area_tag}[/red] [red]❌ {message}[/red]")
        if error is not None:
            for line in str(error).split("\n"):
                console.print(f"{'':25} {line}", style="red")

    @staticmethod
    def warn(area: str, message: str, data=None):
        Logger._log(area, message, data, "yellow", "⚠️")

    @staticmethod
    def debug(area: str, message: str, data=None):
        if os.getenv("DEBUG") == "true":
            Logger._log(area, message, data, "magenta", "🔧")

    @staticmethod
    def database(message: str, data=None):
        Logger._log("database", message, data, "blue", "📊")

    @staticmethod
    def request(method: str, path: str, data=None):
        timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        area_tag = f"[{'REQUEST'}]".ljust(12)
        method_color = {"GET": "green", "POST": "yellow", "PUT": "blue", "DELETE": "red"}.get(method, "white")
        console.print(f"[dim]{timestamp}[/dim] [cyan]{area_tag}[/cyan] [{method_color}]{method}[/{method_color}] [white]{path}[/white]")
        if data is not None:
            console.print(JSON(json.dumps(data), indent=2), style="dim")

    @staticmethod
    def tool(tool_name: str, message: str, data=None):
        timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        area_tag = f"[{'TOOL'}]".ljust(12)
        console.print(f"[dim]{timestamp}[/dim] [magenta]{area_tag}[/magenta] [bold]🔧 {tool_name}[/bold] [white]{message}[/white]")
        if data is not None:
            console.print(JSON(json.dumps(data), indent=2), style="dim")

    @staticmethod
    def separator(title: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")
        line = "═" * 60
        if title:
            padded_title = f" {title} "
            padding = max(0, (len(line) - len(padded_title)) / 2)
            line = f"{'═' * int(padding)}{padded_title}{'═' * int(padding + 0.5)}"
        console.print(f"[dim]{timestamp}[/dim] [cyan]{line}[/cyan]")


# ============================================================================
# Database Tool
# ============================================================================

DB_PATH = "database.db"


class DatabaseToolInput(BaseModel):
    """Input schema for database tool."""

    query: str = Field(..., description="The SQL query to execute")
    params: Optional[List[Any]] = Field(
        default=None,
        description="Optional parameters for prepared statements (prevents SQL injection)",
    )
    mode: str = Field(
        default="query",
        description='Mode: "query" for SELECT/returning data, "exec" for DDL/multiple statements',
    )


def get_db_connection():
    """Get a database connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


async def execute_database_tool(
    query: str, params: Optional[List[Any]] = None, mode: str = "query"
) -> Dict[str, Any]:
    """Execute SQL queries on the SQLite database."""
    params = params or []
    Logger.database(f"Executing {mode.upper()} query", {"query": query[:100] + "..." if len(query) > 100 else query, "params_count": len(params)})
    if params:
        Logger.debug("database", "Query parameters", params)

    start_time = time.time()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Script mode for DDL without params
        if mode == "exec" and not params:
            cursor.executescript(query)
            conn.commit()
            duration = (time.time() - start_time) * 1000
            Logger.success("database", f"Exec completed in {duration:.0f}ms")
            conn.close()
            return {"success": True, "message": "Query executed successfully", "result": None, "duration": duration}

        # Prepared statement mode
        cursor.execute(query, params) if params else cursor.execute(query)
        query_upper = query.strip().upper()
        is_select = query_upper.startswith("SELECT") or "RETURNING" in query_upper or query_upper.startswith("PRAGMA")

        if is_select:
            rows = [dict(row) for row in cursor.fetchall()]
            duration = (time.time() - start_time) * 1000
            Logger.success("database", f"SELECT returned {len(rows)} rows in {duration:.0f}ms")
            conn.close()
            return {"success": True, "rows": rows, "count": len(rows), "duration": duration}
        else:
            conn.commit()
            changes, last_rowid = cursor.rowcount, cursor.lastrowid
            duration = (time.time() - start_time) * 1000
            operation = query.strip().split(" ")[0].upper()
            Logger.success("database", f"{operation} affected {changes} rows in {duration:.0f}ms", {"changes": changes, "last_insert_rowid": last_rowid})
            conn.close()
            return {"success": True, "changes": changes, "last_insert_rowid": last_rowid, "duration": duration}

    except Exception as error:
        duration = (time.time() - start_time) * 1000
        Logger.error("database", f"Query failed after {duration:.0f}ms", error)
        return {"success": False, "error": str(error), "duration": duration}


database_tool = {
    "name": "database",
    "description": "Execute SQL queries on the SQLite database. You can create tables, insert data, query, update, delete - any SQL operation.",
    "input_schema": DatabaseToolInput.model_json_schema(),
    "function": execute_database_tool,
}


# ============================================================================
# Web Response Tool
# ============================================================================


class WebResponseToolInput(BaseModel):
    """Input schema for web response tool."""

    status_code: Optional[int] = Field(
        default=None, description="HTTP status code (default 200)"
    )
    content_type: Optional[str] = Field(
        default=None, description="Content-Type header value"
    )
    body: str = Field(
        ...,
        description="Response body as a string (can be HTML, JSON string, plain text, etc.)",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description='Additional HTTP headers (e.g., {"Location": "/path"} for redirects)',
    )


async def execute_web_response_tool(
    status_code: Optional[int] = None,
    content_type: Optional[str] = None,
    body: str = "",
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate a web response with full control over status, headers, and body."""
    response_headers = headers.copy() if headers else {}

    if content_type:
        response_headers["Content-Type"] = content_type

    return {
        "status_code": status_code or 200,
        "headers": response_headers,
        "body": body,
    }


web_response_tool = {
    "name": "webResponse",
    "description": "Generate a web response with full control over status, headers, and body",
    "input_schema": WebResponseToolInput.model_json_schema(),
    "function": execute_web_response_tool,
}


# ============================================================================
# Memory Tool
# ============================================================================


class MemoryToolInput(BaseModel):
    """Input schema for memory tool."""

    content: str = Field(
        ...,
        description="User preferences, feedback, or instructions to save (markdown format) - these become active directives",
    )
    mode: str = Field(
        ...,
        description="Whether to append to existing memory or rewrite the entire file",
    )


async def execute_memory_tool(content: str, mode: str) -> Dict[str, Any]:
    """Update persistent memory to store user feedback, preferences, and instructions."""
    memory_path = "memory.md"

    try:
        if mode == "append":
            existing_content = ""
            if os.path.exists(memory_path):
                with open(memory_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()
            if existing_content and not existing_content.endswith("\n"):
                existing_content += "\n"
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write(existing_content + content)
            return {"success": True, "message": "Memory appended successfully"}
        else:
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "message": "Memory rewritten successfully"}
    except Exception as error:
        return {"success": False, "message": f"Failed to update memory: {str(error)}"}


memory_tool = {
    "name": "updateMemory",
    "description": "Update persistent memory to store user feedback, preferences, and instructions that shape the application. Memory content becomes active directives for the system. ALWAYS use this for: 1) User feedback about UI/UX preferences, 2) Feature requests, 3) Style preferences, 4) Behavioral changes requested. The memory content is injected into your prompt and becomes part of your instructions.",
    "input_schema": MemoryToolInput.model_json_schema(),
    "function": execute_memory_tool,
}


# ============================================================================
# Tools Registry
# ============================================================================

tools = {
    "database": database_tool,
    "webResponse": web_response_tool,
    "updateMemory": memory_tool,
}


# ============================================================================
# Helper Functions
# ============================================================================


def load_memory() -> str:
    """Load memory content from memory.md file."""
    try:
        if os.path.exists("memory.md"):
            with open("memory.md", "r", encoding="utf-8") as f:
                return f.read()
        return ""
    except Exception as error:
        print(f"Error loading memory.md: {error}")
        return ""


def load_prompt() -> str:
    """Load prompt content from prompt.md file."""
    with open("prompt.md", "r", encoding="utf-8") as f:
        return f.read()


def load_database_schema() -> str:
    """Load and cache database schema on startup."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        tables = cursor.fetchall()

        schema = "\n## DATABASE SCHEMA (Use these exact column names!)\n\n"

        for table in tables:
            if table[0]:
                schema += table[0] + ";\n\n"

        conn.close()
        Logger.success("startup", "Database schema cached for performance")
        return schema
    except Exception as error:
        Logger.error("startup", "Failed to load database schema", error)
        return ""


# Initialize database schema cache
cached_schema = load_database_schema()


# ============================================================================
# Helper Functions
# ============================================================================


def log_print(message: str):
    """Print with timestamp prefix."""
    print(f"[{datetime.now().isoformat()}] {message}")


# ============================================================================
# Tool Execution
# ============================================================================


async def execute_tool(tool_name: str, args: dict, request_id: str, step: int) -> dict:
    """Execute a tool and return its result."""
    Logger.tool(tool_name, "called", {"request_id": request_id, "step": step, "args": list(args.keys())})

    result = None
    if tool_name == "database":
        result = await execute_database_tool(**args)
    elif tool_name == "webResponse":
        result = await execute_web_response_tool(**args)
    elif tool_name == "updateMemory":
        result = await execute_memory_tool(**args)

    result_size = len(json.dumps(result)) if result else 0
    Logger.tool(tool_name, f"completed ({result_size} chars)", {
        "request_id": request_id, "step": step,
        "success": result.get("success", True) if isinstance(result, dict) else True
    })
    return result


# ============================================================================
# LLM Handlers
# ============================================================================


async def handle_anthropic_request(
    request_context: Dict[str, Any], prompt: str, request_id: str
) -> Dict[str, Any]:
    """Handle request using Anthropic Claude with streaming."""
    client = Anthropic(api_key=config.anthropic.api_key)

    # Convert tools to Anthropic format
    anthropic_tools = []
    for tool_name, tool_def in tools.items():
        anthropic_tools.append(
            {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "input_schema": tool_def["input_schema"],
            }
        )

    log_print("Starting LLM call (streaming)...")
    llm_start_time = time.time()

    steps = []
    step_count = 0
    max_steps = 10
    messages = [{"role": "user", "content": prompt}]

    stop_reason = None
    final_text = ""
    assistant_content = []

    # Initial streaming request
    while step_count == 0 or (stop_reason == "tool_use" and step_count < max_steps):
        if step_count > 0:
            step_time = (time.time() - llm_start_time) * 1000
            log_print(f"Step {step_count} completed at {step_time:.0f}ms")

        step_count += 1
        current_text = ""
        tool_calls = []
        current_tool = None

        # Stream the response
        with client.messages.stream(
            model=config.anthropic.model,
            max_tokens=50000,
            tools=anthropic_tools,
            messages=messages,
            thinking={"type": "disabled"},
        ) as stream:
            for event in stream:
                # Handle text delta
                if hasattr(event, "type") and event.type == "content_block_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "type"):
                        if event.delta.type == "text_delta":
                            chunk = event.delta.text
                            current_text += chunk
                            print(chunk, end="", flush=True)
                        elif event.delta.type == "input_json_delta":
                            if current_tool:
                                current_tool["partial_json"] += event.delta.partial_json

                # Handle content block start
                elif hasattr(event, "type") and event.type == "content_block_start":
                    if hasattr(event, "content_block"):
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "partial_json": "",
                            }
                            log_print(f"\nTool call started: {event.content_block.name}")

                # Handle content block stop
                elif hasattr(event, "type") and event.type == "content_block_stop":
                    if current_tool:
                        try:
                            tool_input = json.loads(current_tool["partial_json"])
                            tool_calls.append(
                                {
                                    "tool_name": current_tool["name"],
                                    "args": tool_input,
                                    "id": current_tool["id"],
                                }
                            )
                            log_print(f"\n  - Tool: {current_tool['name']} ({len(current_tool['partial_json'])} chars args)")
                        except json.JSONDecodeError as e:
                            log_print(f"\nError parsing tool input: {e}")
                        current_tool = None

            # Get final message
            final_message = stream.get_final_message()
            stop_reason = final_message.stop_reason
            assistant_content = final_message.content

        if current_text:
            print()  # New line after streaming text

        final_text = current_text

        # If no tool calls, we're done
        if stop_reason != "tool_use" or not tool_calls:
            break

        # Execute tools
        tool_results = []
        for tc in tool_calls:
            result = await execute_tool(tc["tool_name"], tc["args"], request_id, step_count)
            tool_results.append({"tool_name": tc["tool_name"], "result": result, "tool_use_id": tc["id"]})

        steps.append({"tool_calls": tool_calls, "tool_results": tool_results})

        # Continue conversation with tool results
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]

        # Add tool results
        tool_result_content = []
        for tr in tool_results:
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tr["tool_use_id"],
                    "content": json.dumps(tr["result"]),
                }
            )

        messages.append({"role": "user", "content": tool_result_content})

    llm_duration = (time.time() - llm_start_time) * 1000
    log_print(f"LLM call completed in {llm_duration:.0f}ms")

    return {"steps": steps, "text": final_text, "llm_duration": llm_duration}


async def handle_openai_request(
    request_context: Dict[str, Any], prompt: str, request_id: str
) -> Dict[str, Any]:
    """Handle request using OpenAI with streaming."""
    client = OpenAI(api_key=config.openai.api_key)

    # Convert tools to OpenAI format
    openai_tools = []
    for tool_name, tool_def in tools.items():
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": tool_def["input_schema"],
                },
            }
        )

    log_print("Starting LLM call (streaming)...")
    llm_start_time = time.time()

    messages = [{"role": "user", "content": prompt}]
    steps = []
    step_count = 0
    max_steps = 10
    finish_reason = None
    final_text = ""

    # Process streaming requests in a loop
    while step_count == 0 or (finish_reason == "tool_calls" and step_count < max_steps):
        if step_count > 0:
            step_time = (time.time() - llm_start_time) * 1000
            log_print(f"Step {step_count} completed at {step_time:.0f}ms")

        step_count += 1
        current_text = ""
        current_tool_calls = {}

        # Stream the response
        stream = client.chat.completions.create(
            model=config.openai.model,
            messages=messages,
            tools=openai_tools,
            max_tokens=8000,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # Handle text content
            if delta.content:
                current_text += delta.content
                print(delta.content, end="", flush=True)

            # Handle tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tc_delta.id:
                        current_tool_calls[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            current_tool_calls[idx]["name"] = tc_delta.function.name
                            log_print(f"\nTool call started: {tc_delta.function.name}")
                        if tc_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += (
                                tc_delta.function.arguments
                            )

            # Get finish reason from chunk
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        if current_text:
            print()  # New line after streaming text

        final_text = current_text

        # Process completed tool calls
        tool_calls = []
        tool_results = []

        if finish_reason == "tool_calls" and current_tool_calls:
            for idx in sorted(current_tool_calls.keys()):
                tc = current_tool_calls[idx]
                try:
                    args = json.loads(tc["arguments"])
                    tool_name = tc["name"]

                    tool_calls.append(
                        {"tool_name": tool_name, "args": args, "id": tc["id"]}
                    )
                    log_print(f"  - Tool: {tool_name} ({len(tc['arguments'])} chars args)")

                    result = await execute_tool(tool_name, args, request_id, step_count)
                    tool_results.append({"tool_name": tool_name, "result": result, "tool_call_id": tc["id"]})

                except json.JSONDecodeError as e:
                    log_print(f"Error parsing tool arguments: {e}")

            # Add assistant message with tool calls to conversation
            assistant_tool_calls = []
            for idx in sorted(current_tool_calls.keys()):
                tc = current_tool_calls[idx]
                assistant_tool_calls.append(
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": current_text if current_text else None,
                    "tool_calls": assistant_tool_calls,
                }
            )

            # Add tool results to messages
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": json.dumps(tr["result"]),
                    }
                )

            steps.append({"tool_calls": tool_calls, "tool_results": tool_results})
        else:
            # No more tool calls, we're done
            break

    llm_duration = (time.time() - llm_start_time) * 1000
    log_print(f"LLM call completed in {llm_duration:.0f}ms")

    return {"steps": steps, "text": final_text, "llm_duration": llm_duration}


# ============================================================================
# Main Request Handler
# ============================================================================


async def handle_llm_request(request: Request):
    """Main LLM request handler middleware."""
    request_start_time = time.time()
    request_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))

    # Get request body - handle both JSON and form-encoded data
    body = {}
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            body = {k: v for k, v in (await request.form()).items()}
        elif request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
            except Exception:
                try:
                    body = {k: v for k, v in (await request.form()).items()}
                except Exception:
                    pass
    except Exception as e:
        log_print(f"Warning: Could not parse request body: {e}")

    path = request.url.path
    log_print(f"=== REQUEST START: {request.method} {path} ===")

    try:
        # Enhanced request logging
        Logger.request(
            request.method,
            path,
            {
                "request_id": request_id,
                "query": dict(request.query_params),
                "body_size": len(json.dumps(body)),
                "user_agent": request.headers.get("user-agent", "")[:50] + "...",
                "ip": request.client.host if request.client else "unknown",
            },
        )

        # Prepare request context
        request_context = {
            "method": request.method,
            "path": path,
            "query": dict(request.query_params),
            "headers": dict(request.headers),
            "body": body,
            "url": str(request.url),
            "ip": request.client.host if request.client else "unknown",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
        }

        # Load memory and prompt
        log_print("Loading memory and prompt...")
        memory_start_time = time.time()
        memory = load_memory()
        system_prompt_template = load_prompt()
        memory_duration = (time.time() - memory_start_time) * 1000
        log_print(f"Memory and prompt loaded in {memory_duration:.0f}ms")

        # Pre-fetch database context
        database_context = ""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM books")
            result = cursor.fetchone()
            conn.close()
            if result and result[0] > 0:
                database_context = f"\n## DATABASE CONTEXT\n\nThe database currently contains {result[0]} book(s). Use the database tool to query them if needed for this request.\n\n"
        except Exception:
            pass

        # Replace template variables
        prompt = (
            system_prompt_template.replace("{{METHOD}}", request_context["method"])
            .replace("{{PATH}}", request_context["path"])
            .replace("{{URL}}", request_context["url"])
            .replace("{{QUERY}}", json.dumps(request_context["query"]))
            .replace("{{HEADERS}}", json.dumps(request_context["headers"]))
            .replace("{{BODY}}", json.dumps(request_context["body"]))
            .replace("{{IP}}", request_context["ip"])
            .replace("{{TIMESTAMP}}", request_context["timestamp"])
            .replace("{{MEMORY}}", memory + cached_schema + database_context)
        )

        # Log model selection
        Logger.info(
            "llm",
            f"Using {config.provider} provider with model {config.anthropic.model if config.provider == 'anthropic' else config.openai.model}",
            {"request_id": request_id, "provider": config.provider},
        )

        # Debug logging
        log_print("=== PROMPT ANALYSIS ===")
        log_print(f"Request: {request.method} {path}")
        log_print(f"Prompt size: {len(prompt)} characters")
        log_print(f"Memory size: {len(memory)} characters")
        log_print(f"Schema size: {len(cached_schema)} characters")

        is_api_request = path.startswith("/api/")
        expects_html = not is_api_request and request.method == "GET"
        log_print(f"Request type: {'API (JSON)' if is_api_request else 'HTML Page' if expects_html else 'Other'}")

        # Call appropriate LLM
        if config.provider == "openai":
            result = await handle_openai_request(request_context, prompt, request_id)
        else:
            result = await handle_anthropic_request(request_context, prompt, request_id)

        llm_duration = result["llm_duration"]

        # Enhanced step-by-step logging
        Logger.separator(f"LLM EXECUTION COMPLETE ({llm_duration:.0f}ms)")
        Logger.info(
            "llm",
            f"Request {request.method} {path} completed",
            {
                "request_id": request_id,
                "total_steps": len(result.get("steps", [])),
                "llm_duration": llm_duration,
                "has_body": bool(request_context["body"]),
            },
        )

        if result.get("steps"):
            log_print(f"Processing {len(result['steps'])} tool steps...")
            for idx, step in enumerate(result["steps"]):
                Logger.debug("llm", f"Step {idx + 1} execution")
                for tc in step.get("tool_calls", []):
                    log_print(f"Tool call: {tc['tool_name']}")
                for tr in step.get("tool_results", []):
                    log_print(f"Tool {tr['tool_name']} completed")

        Logger.separator()

        # Look for webResponse tool across ALL steps
        web_response_result = None
        last_tool_result = None

        if result.get("steps"):
            for step in result["steps"]:
                if step.get("tool_results"):
                    last_tool_result = step["tool_results"][-1]
                    web_response = next(
                        (
                            tr
                            for tr in step["tool_results"]
                            if tr["tool_name"] == "webResponse"
                        ),
                        None,
                    )
                    if web_response:
                        web_response_result = web_response
                        break

        # Process the response
        total_request_duration = (time.time() - request_start_time) * 1000
        log_print(f"Preparing response after {total_request_duration:.0f}ms...")

        if web_response_result:
            output = web_response_result["result"]
            log_print(f"Sending webResponse with status {output.get('status_code', 200)}")
            Logger.success(
                "response",
                f"Sending webResponse ({output.get('status_code', 200)})",
                {
                    "request_id": request_id,
                    "status_code": output.get("status_code", 200),
                    "body_size": len(output.get("body", "")),
                    "has_headers": bool(output.get("headers")),
                    "total_duration": total_request_duration,
                },
            )

            headers = output.get("headers", {})
            log_print(f"=== REQUEST COMPLETE in {total_request_duration:.0f}ms ===")

            return Response(
                content=output.get("body", ""),
                status_code=output.get("status_code", 200),
                headers=headers,
            )

        elif last_tool_result:
            output = last_tool_result["result"]
            Logger.warn(
                "response",
                f"No webResponse found, using {last_tool_result['tool_name']} output as fallback",
                {
                    "request_id": request_id,
                    "tool_name": last_tool_result["tool_name"],
                    "output_size": len(json.dumps(output)),
                    "total_duration": total_request_duration,
                },
            )

            return JSONResponse(content=output)

        else:
            Logger.warn(
                "response",
                "No tools called, returning text response",
                {
                    "request_id": request_id,
                    "text_length": len(result.get("text", "")),
                    "total_duration": total_request_duration,
                },
            )
            return Response(content=result.get("text", "No response generated"))

    except Exception as error:
        total_request_duration = (time.time() - request_start_time) * 1000
        Logger.error(
            "request",
            f"Request failed after {total_request_duration:.0f}ms",
            {
                "request_id": request_id,
                "error": str(error),
                "method": request.method,
                "path": path,
            },
        )

        return Response(
            content=f"""
        <html>
            <body>
                <h1>Server Error</h1>
                <p>An error occurred while processing your request.</p>
                <p><strong>Request ID:</strong> {request_id}</p>
                <pre>{str(error)}</pre>
            </body>
        </html>
        """,
            status_code=500,
            media_type="text/html",
        )


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="LLM-JIT", description="A web server where AI handles all requests")


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
)
async def catch_all(request: Request, path: str = ""):
    """Catch all routes and handle with LLM."""
    return await handle_llm_request(request)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Start the server."""
    print(f"🤖 Server running on http://localhost:{config.port}")
    print(f"🧠 Using {config.provider} provider")

    model = (
        config.anthropic.model
        if config.provider == "anthropic"
        else config.openai.model
    )
    print(f"⚡ Model: {model}")

    print(
        "🚀 Every request will be handled by AI. Make any HTTP request and see what happens."
    )
    print("💰 Warning: Each request costs API tokens!")

    uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")


if __name__ == "__main__":
    main()
