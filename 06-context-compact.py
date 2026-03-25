#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import dashscope
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
logging.basicConfig(level=logging.INFO, filename=str(WORKDIR / "agent.log"))

MODEL = os.environ.get("MODEL_ID", "default_model")

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    tool_messages = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_messages.append((msg_idx, msg))

    if len(tool_messages) <= KEEP_RECENT:
        return messages

    tool_name_map = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls", []) or []:
                tool_name_map[tool_call.get("id")] = tool_call["function"]["name"]

    for _, tool_msg in tool_messages[:-KEEP_RECENT]:
        content = tool_msg.get("content")
        if isinstance(content, str) and len(content) > 100:
            tool_id = tool_msg.get("tool_call_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            tool_msg["content"] = f"[Previous: used {tool_name}]"

    return messages


def summarize_conversation(messages: list, transcript_path: Path) -> str:
    conversation_text = json.dumps(messages, ensure_ascii=False, default=str)[:80000]
    response = dashscope.Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this conversation for continuity. Include: "
                    "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                    "Be concise but preserve critical details.\n\n"
                    f"Transcript file: {transcript_path}\n\n"
                    f"{conversation_text}"
                ),
            }
        ],
        result_format="message",
    )
    assistant_output = response.output.choices[0].message
    return assistant_output.get("content", "") or "(no summary)"


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    summary = summarize_conversation(messages, transcript_path)
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
        {
            "role": "assistant",
            "content": "Understood. I have the context from the summary. Continuing.",
        },
    ]


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text(encoding="utf-8")
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_tool(func_name: str, arguments: dict) -> str:
    if func_name == "bash":
        return run_bash(arguments["command"])
    if func_name == "read":
        return run_read(arguments["path"], arguments.get("limit"))
    if func_name == "write":
        return run_write(arguments["path"], arguments["content"])
    if func_name == "edit":
        return run_edit(arguments["path"], arguments["old_text"], arguments["new_text"])
    if func_name == "compact":
        return "Manual compression requested."
    return f"Error: Unknown tool: {func_name}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary",
                    }
                },
            },
        },
    },
]


def agent_loop(messages: list):
    while True:
        micro_compact(messages)

        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        response = dashscope.Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=MODEL,
            messages=messages,
            result_format="message",
            tools=TOOLS,
        )

        assistant_output = response.output.choices[0].message
        messages.append(assistant_output)

        if "tool_calls" not in assistant_output or not assistant_output["tool_calls"]:
            return

        tool_call = assistant_output["tool_calls"][0]
        func_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call.get("id")
        logging.info(f"正在调用工具 [{func_name}]，参数：{arguments}")

        manual_compact = func_name == "compact"
        try:
            output = run_tool(func_name, arguments)
            logging.info(str(output)[:200])
        except Exception as e:
            logging.error(f"Error: {e}")
            output = f"Error: {e}"

        print(f"> {func_name}: {str(output)[:200]}")
        messages.append(
            {
                "role": "tool",
                "content": str(output),
                "tool_call_id": tool_call_id,
            }
        )

        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
            logging.info(f"s06:{query}")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, str) and response_content:
            print(response_content)
        print()
