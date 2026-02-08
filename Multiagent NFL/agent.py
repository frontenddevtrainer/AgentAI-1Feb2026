from __future__ import annotations

import json
import os
import re
import ssl
import operator
import urllib.error

from html.parser import HTMLParser
from typing import Any, Annotated, TypedDict
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import ToolNode


from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
DEFAULT_SEARCH_RESULTS = 5
DEFAULT_SCRAPE_CHARS = 3500
WEB_VERIFY_SSL = os.getenv("WEB_VERIFY_SSL", "true").lower() in {"1", "true", "yes", "y"}

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8000").rstrip("/")
MCP_TIMEOUT_SECONDS = float(os.getenv("MCP_TIMEOUT_SECONDS", "15"))
MCP_VERIFY_SSL = os.getenv("MCP_VERIFY_SSL", "false").lower() in {"1", "true", "yes", "y"}

class NflAgentState(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    final_answer: str

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignore_depth:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        text = data.strip()
        if text:
            self._texts.append(text)

    def get_text(self) -> str:
        return " ".join(self._texts)


class _DuckDuckGoParser(HTMLParser):
    def __init__(self, max_results: int) -> None:
        super().__init__()
        self.max_results = max_results
        self.results: list[dict[str, str]] = []
        self._capture = False
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        class_value = attr_map.get("class", "")
        if "result__a" not in class_value:
            return
        self._capture = True
        self._current_href = attr_map.get("href")
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture:
            return
        title = "".join(self._text_parts).strip()
        href = self._current_href or ""
        if title and href and len(self.results) < self.max_results:
            self.results.append({"title": title, "url": _clean_ddg_url(href)})
        self._capture = False
        self._current_href = None
        self._text_parts = []

def _clean_ddg_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
    return raw_url



def _fetch_url(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = None
    if url.startswith("https") and not WEB_VERIFY_SSL:
        context = ssl._create_unverified_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8", errors="ignore")


@tool("web_search")
def web_search(query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> str:
    """
    Perform a lightweight web search using DuckDuckGo and return parsed results.

    This tool sends the query to DuckDuckGo's HTML endpoint, parses the
    search results page, and returns a JSON string containing the original
    query and a list of search results.

    Args:
        query (str): The search query to look up on the web.
        max_result (int): Maximum number of search results to return.
            Defaults to DEFAULT_SEARCH_RESULTS.

    Returns:
        str: A JSON-encoded string with the following structure:
            {
                "query": "<original query>",
                "results": [
                    {
                        "title": "<result title>",
                        "url": "<result URL>",
                        "snippet": "<short description>"
                    },
                    ...
                ]
            }

    Notes:
        - Uses DuckDuckGo's public HTML interface (no API key required).
        - Results are best-effort and may change if the page structure changes.
        - Intended for general information lookup, not guaranteed real-time accuracy.
    """
    encoded = urlencode({"q": query})
    url = f"https://duckduckgo.com/html/?{encoded}"
    html = _fetch_url(url, timeout=15)
    parser = _DuckDuckGoParser(max_results=max_results)
    parser.feed(html)
    payload = {"query": query, "results": parser.results}
    print(payload)
    return json.dumps(payload, ensure_ascii=True)


@tool("web_scrape")
def web_scrape(url: str, max_chars: int = DEFAULT_SCRAPE_CHARS) -> str:
    """
    Scrapes a webpage and returns the extracted text in JSON format.

    Parameters:
    - url (str): The URL of the webpage to scrape.
    - max_chars (int, optional): The maximum number of characters to extract from the webpage's text.
      If the extracted text exceeds this length, it will be truncated. Default is `DEFAULT_SCRAPE_CHARS`.

    Returns:
    - str: A JSON-encoded string containing the following:
        - "url": The URL of the page that was scraped.
        - "content": The extracted text from the webpage, truncated to the specified `max_chars` length.

    This function fetches the content of the given URL, extracts the text from the HTML, and cleans it by
    removing extra whitespace. If the text length exceeds the specified `max_chars`, it will be truncated
    with "..." appended to indicate more content. If the page is blocked (HTTP 403 error), a message is returned.
    """
    try:
        html = _fetch_url(url, timeout=20)
        parser = _TextExtractor()
        parser.feed(html)
        text = re.sub(r"\s+", " ", parser.get_text()).strip()

        # Truncate text if it exceeds the specified max_chars
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        payload = {"url": url, "content": text}
        return json.dumps(payload, ensure_ascii=True)

    except urllib.error.HTTPError as e:
        if e.code == 403:
            # Handle HTTP 403: Forbidden
            error_message = {
                "error": "HTTP Error 403: Forbidden",
                "message": "The website blocked your request. Please ensure the page is publicly accessible and try again."
            }
            return json.dumps(error_message, ensure_ascii=True)

    except Exception as e:
        # Handle other errors like network issues, etc.
        error_message = {
            "error": str(e),
            "message": "An error occurred while scraping the webpage."
        }
        return json.dumps(error_message, ensure_ascii=True)

@tool("mcp_nfl_query")
def mcp_nfl_query(endpoint: str, params: dict | None = None) -> str:
    """
    Query the MCP server for NFL data.

    This tool retrieves structured mock NFL data from an MCP-compatible
    backend service. It should be preferred over web_search and web_scrape
    whenever MCP data is available.

    Args:
        endpoint (str): MCP API endpoint (e.g. "/passing-leaders").
        params (dict, optional): Query parameters to send with the request.

    Returns:
        str: JSON-encoded response from the MCP server.
    """
    params = params or {}
    query = urlencode(params)
    url = f"{MCP_BASE_URL}{endpoint}"
    if query:
        url += f"?{query}"

    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = None
    if url.startswith("https") and not MCP_VERIFY_SSL:
        context = ssl._create_unverified_context()

    with urlopen(request, timeout=MCP_TIMEOUT_SECONDS, context=context) as response:
        payload = response.read().decode("utf-8")
        return payload

@tool("current_datetime")
def current_datetime(tz: str = "UTC", iso: bool = True) -> str:
    """
    Returns the current date and time in the specified timezone.

    Parameters:
    - tz (str): The timezone for the current time. Options are 'UTC' or 'local'. Default is 'UTC'.
    - iso (bool): If True, returns the datetime in ISO 8601 format. If False, returns the datetime in a more human-readable format. Default is True.

    Returns:
    - str: A JSON-encoded string with the current date and time, including the timezone.
      The JSON includes the following keys:
      - "timezone": The name of the timezone (e.g., "UTC" or the local timezone).
      - "datetime": The full datetime in the specified format (ISO or custom).
      - "date": The current date in ISO format (YYYY-MM-DD).
      - "time": The current time in HH:MM:SS format.
    """
    if tz.lower() == "local":
        now = datetime.now().astimezone()
        tz_name = str(now.tzinfo)
    else:
        now = datetime.now(timezone.utc)
        tz_name = "UTC"

    payload = {
        "timezone": tz_name,
        "datetime": now.isoformat() if iso else now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.date().isoformat(),
        "time": now.time().strftime("%H:%M:%S"),
    }

    return json.dumps(payload, ensure_ascii=True)

def build_agent() -> any:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the NFL multi-agent graph.")
    
    model = ChatOpenAI(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )

    tools = [
        web_search,
        web_scrape,
        # mcp_nfl_query,
        current_datetime
    ]

    model_with_tools = model.bind_tools(tools)

    system_prompt = (
        """
            You are an NFL news and statistics analyst.

            You MUST follow this workflow exactly:

            1. Always call the current_datetime tool first to determine today’s date.
            2. If the question requires factual, ranked, or numeric information
            (such as leaders, stats, standings, injuries, or depth charts),
            you MUST call web_search to find authoritative sources.
            3. Only if MCP data is NOT available may you fall back to web_search
            followed by web_scrape.
            4. If web_search is called, you MUST extract facts from at least one
            authoritative source by calling web_scrape before answering.
            Never answer using links alone.
            5. Prefer official or authoritative sources in this order:
            - Pro-Football-Reference
            - NFL.com
            - ESPN
            - StatMuse
            6. Only provide a final answer AFTER factual data has been extracted
            via web_scrape.
            7. If scraping fails, clearly state that the data could not be retrieved
            and explain why.

            Answer guidelines:
            - Be concise and factual.
            - State the date the information applies to.
            - Summarize key findings in bullet points when possible.
            - Include a final “Sources” line listing scraped URLs.

            Do NOT guess, infer, or defer the user to external links.
            Do NOT provide an answer without scraping when facts are required.

        """
    )

    def agent_node(state: NflAgentState) -> NflAgentState:
        messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def finalize(state: NflAgentState) -> NflAgentState:
        messages = state.get("messages") or []
        if messages and isinstance(messages[-1], AIMessage):
            return {"final_answer": messages[-1].content.strip()}
        return {"final_answer": ""}
    
    def should_continue(state: NflAgentState) -> str:
        messages = state.get("messages") or []
        if not messages:
            return "finalize"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "finalize"

    graph = StateGraph(NflAgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finalize", finalize)    

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()

def answer_question(question: str) -> str:
    graph = build_agent()
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return result.get("final_answer", "").strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFL multi-agent tool runner.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Who is the passing leader?",
        help="Question for the multi-agent system.",
    )
    args = parser.parse_args()
    print(answer_question(args.question))


