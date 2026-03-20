import os
import subprocess
import dashscope
import json
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO,
filename="agent.log")

WORKDIR = Path.cwd()

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at{sys.platform} {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

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
                "properties": {"path": {"type": "string"}},
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
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
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
                "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    # delete
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "Delete a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
        }
    }
]


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
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

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

def run_delete(path: str) -> str:
    try:
        fp = safe_path(path)
        fp.unlink(missing_ok=True)
        return f"Deleted {path}"
    except Exception as e:
        return f"Error: {e}"

def run_tool(func_name: str, arguments: dict) -> str:
    if func_name == "bash":
        return run_bash(arguments["command"])
    elif func_name == "read":
        return run_read(arguments["path"])
    elif func_name == "write":
        return run_write(arguments["path"], arguments["content"])
    elif func_name == "edit":
        return run_edit(arguments["path"], arguments["old_text"], arguments["new_text"])
    elif func_name == "delete":
        return run_delete(arguments["path"])
    else:
        return f"Error: Unknown tool: {func_name}"

# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = dashscope.Generation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=MODEL, # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            result_format='message',
            tools=TOOLS
            )

        assistant_output = response.output.choices[0].message
        # Append assistant turn
        messages.append(assistant_output)
        # If the model didn't call a tool, we're done
        if "tool_calls" not in assistant_output or not assistant_output["tool_calls"]:
            return

        # Execute each tool call, collect results
        tool_call = assistant_output["tool_calls"][0]
        # 解析工具调用的信息
        func_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call.get("id")  # 获取 tool_call_id
        logging.info(f"正在调用工具 [{func_name}]，参数：{arguments}")
        try:
            output = run_tool(func_name, arguments)
            logging.info(output[:200])
        except Exception as e:
            logging.error(f"Error: {e}")
            output = f"Error: {e}"

        tool_message = {
            "role": "tool",
            "content": output,
            "tool_call_id": tool_call_id
        }
        messages.append(tool_message)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
            logging.info(f"s01:{query}")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                logging.info(block)
        else:
            logging.info(response_content)