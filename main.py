#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess
import dashscope
import json
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

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
    }
]


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


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
        # print(response)
        # print(messages)
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
        print(f"正在调用工具 [{func_name}]，参数：{arguments}")
        output = run_bash(arguments["command"])
        print(output[:200])

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
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                print(block)
        else:
            print(response_content)


""" 
# 未添加tool工具调用
s01 >> 今天日期
今天是2023年10月5日。
s01 >> 今日添加
正在调用工具 [bash]，参数：{'command': 'date'}
2026年 3月20日 星期五 11时14分06秒 CST
今天是2026年3月20日。
s01 >> 今日上海天气
正在调用工具 [bash]，参数：{'command': "curl -s 'http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q=Shanghai' | jq '.current.condition.text'"}
/bin/sh: jq: command not found
抱歉，我无法直接查询天气信息。你可以访问天气预报网站或使用天气应用程序来获取最新的天气情况。
"""