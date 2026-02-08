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

SYSTEM_PROMPT = """You are a helpful AI assistant with access to three tools:

1. **web_search** — Search the internet for up-to-date information using DuckDuckGo.
2. **calculator** — Evaluate mathematical expressions (supports standard math operators and functions like sqrt, sin, cos, log, pi, e).
3. **python_repl** — Execute Python code and return the output. Use this for tasks that need programming, data manipulation, or anything beyond simple math.

Guidelines:
- Pick the most appropriate tool for each sub-task.
- For math, prefer the calculator. For complex logic or multi-step computation, use python_repl.
- Always explain your reasoning before and after using tools.
- If a tool call fails, try an alternative approach.
"""


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

def create_agent():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.3, api_key=api_key)

    tools = [calculator]

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
    response = run_agent(agent, "What is the square root of 1764 plus 42?")
    print(f"Agent: {response}")

if __name__ == "__main__":
    run_demo()