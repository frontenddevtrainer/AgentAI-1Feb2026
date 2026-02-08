import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are a helpful AI assistant with access to five tools:

when user ask question which dates and time check the current date time and answer using that knowledge

1. **web_search** — Search the internet for up-to-date information using DuckDuckGo.
2. **calculator** — Evaluate mathematical expressions (supports standard math operators and functions like sqrt, sin, cos, log, pi, e).
3. **python_repl** — Execute Python code and return the output. Use this for tasks that need programming, data manipulation, or anything beyond simple math.
4. **get_current_datetime** — Get the current date and time in any timezone.
5. **convert_time** — Convert a time from one timezone to another.

Guidelines:
- Pick the most appropriate tool for each sub-task.
- For math, prefer the calculator. For complex logic or multi-step computation, use python_repl.
- For date/time questions, use get_current_datetime or convert_time.
- Always explain your reasoning before and after using tools.
- If a tool call fails, try an alternative approach.
"""

search_tool = DuckDuckGoSearchRun(name="web_search")

python_repl = PythonREPLTool(name="python_repl")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math expression such as "2**10", "sqrt(144)", "sin(pi/4)"
    """

    try:
        import numexpr
        result = numexpr.evaluate(expression).item()
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"
    
@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time in a given timezone.

    Args:
        timezone: IANA timezone name, e.g. "UTC", "US/Eastern", "Europe/London", "Asia/Tokyo"
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return (
            f"Current date/time in {timezone}:\n"
            f"  Date: {now.strftime('%Y-%m-%d (%A)')}\n"
            f"  Time: {now.strftime('%H:%M:%S %Z')}"
        )
    except Exception as e:
        return f"Error: invalid timezone '{timezone}' — {e}"
    
@tool
def convert_time(time_str: str, from_tz: str, to_tz: str) -> str:
    """Convert a time from one timezone to another.

    Args:
        time_str: Time in HH:MM (24-hour) format, e.g. "14:30"
        from_tz: Source IANA timezone, e.g. "US/Eastern"
        to_tz: Target IANA timezone, e.g. "Asia/Tokyo"
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        src = ZoneInfo(from_tz)
        dst = ZoneInfo(to_tz)
        today = datetime.now(src).date()
        dt = datetime.strptime(time_str, "%H:%M").replace(
            year=today.year, month=today.month, day=today.day, tzinfo=src
        )
        converted = dt.astimezone(dst)
        return (
            f"{time_str} {from_tz} = {converted.strftime('%H:%M')} {to_tz} "
            f"({converted.strftime('%Y-%m-%d %Z')})"
        )
    except Exception as e:
        return f"Error converting time: {e}"

def create_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.3, api_key=api_key)

    tools = [calculator, search_tool, convert_time, get_current_datetime, python_repl]

    agent = create_react_agent(
        model = llm,
        tools = tools,
        prompt=SYSTEM_PROMPT
    )

    return agent


def run_agent(agent, query: str) -> str:
    config = {"recursion_limit": 50}
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return result["messages"][-1].content

def run_demo():
    agent = create_agent()
    response = run_agent(agent, "Write a Python script that generates the first 10 Fibonacci numbers and prints them.")
    print(f"Agent: {response}")

if __name__ == "__main__":
    run_demo()