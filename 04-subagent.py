#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import json
import logging
import os
import subprocess
from pathlib import Path

import dashscope
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
logging.basicConfig(level=logging.INFO, filename=str(WORKDIR / "agent.log"))

MODEL = os.environ.get("MODEL_ID", "default_model")

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use the task tool to delegate exploration or subtasks."
)
SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the given task, then summarize your findings."
)


# -- Tool implementations shared by parent and child --
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
    return f"Error: Unknown tool: {func_name}"


BASE_TOOLS = [
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
]

CHILD_TOOLS = BASE_TOOLS
PARENT_TOOLS = BASE_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": (
                "Spawn a subagent with fresh context. "
                "It shares the filesystem but not conversation history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    }
]


def call_model(messages: list, system_prompt: str, tools: list) -> dict:
    response = dashscope.Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        result_format="message",
        tools=tools,
    )
    return response.output.choices[0].message


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        assistant_output = call_model(sub_messages, SUBAGENT_SYSTEM, CHILD_TOOLS)
        sub_messages.append(assistant_output)

        tool_calls = assistant_output.get("tool_calls") or []
        if not tool_calls:
            content = assistant_output.get("content", "")
            return content if content else "(no summary)"

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call.get("id")
            logging.info(f"subagent tool [{func_name}] args={arguments}")
            try:
                output = run_tool(func_name, arguments)
            except Exception as e:
                logging.error(f"subagent tool error: {e}")
                output = f"Error: {e}"
            logging.info(str(output)[:200])
            sub_messages.append(
                {
                    "role": "tool",
                    "content": str(output),
                    "tool_call_id": tool_call_id,
                }
            )
    return "Error: Subagent exceeded 30 rounds"


def agent_loop(messages: list):
    while True:
        assistant_output = call_model(messages, SYSTEM, PARENT_TOOLS)
        messages.append(assistant_output)

        tool_calls = assistant_output.get("tool_calls") or []
        if not tool_calls:
            return

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call.get("id")
            logging.info(f"parent tool [{func_name}] args={arguments}")

            try:
                if func_name == "task":
                    desc = arguments.get("description", "subtask")
                    print(f"> task ({desc}): {arguments['prompt'][:80]}")
                    output = run_subagent(arguments["prompt"])
                else:
                    output = run_tool(func_name, arguments)
            except Exception as e:
                logging.error(f"parent tool error: {e}")
                output = f"Error: {e}"

            print(f"  {str(output)[:200]}")
            logging.info(str(output)[:200])
            messages.append(
                {
                    "role": "tool",
                    "content": str(output),
                    "tool_call_id": tool_call_id,
                }
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
            logging.info(f"s04: {query}")
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
