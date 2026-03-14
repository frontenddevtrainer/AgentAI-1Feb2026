from autogen_agentchat.base import ChatAgent

SELECTOR_PROMPT = """You are orchestrating an AI content creation pipeline. Select the NEXT agent to speak.

Pipeline order: IdeationAgent -> OutlineAgent -> WriterAgent -> SEOAgent -> ImagePromptAgent -> PublisherAgent

Rules:
- If no agent has spoken yet, select IdeationAgent.
- If the last message contains IDEATION_COMPLETE, select OutlineAgent.
- If the last message contains OUTLINE_COMPLETE, select WriterAgent.
- If the last message contains DRAFT_COMPLETE, select SEOAgent.
- If the last message contains SEO_COMPLETE, select ImagePromptAgent.
- If the last message contains IMAGES_COMPLETE, select PublisherAgent.
- If the last message contains PUBLISH_READY or TERMINATE, do NOT select any agent.
- If the user asks for revisions to a specific stage, select the appropriate upstream agent.

Available agents: {participants}
Agent roles:
{roles}

Conversation history:
{history}

Reply with ONLY the agent name. No explanation.
"""

ROUTING_MAP = {
    "IDEATION_COMPLETE": "OutlineAgent",
    "OUTLINE_COMPLETE": "WriterAgent",
    "DRAFT_COMPLETE": "SEOAgent",
    "SEO_COMPLETE": "ImagePromptAgent",
    "IMAGES_COMPLETE": "PublisherAgent",
}


def selector_func(messages: list) -> str | None:
    """
    Fast-path router based on terminal keywords in the last message.
    Returns None to fall back to LLM-based selection (e.g. for revision requests).
    """
    if not messages:
        return "IdeationAgent"

    last = messages[-1]
    content = getattr(last, "content", "") or ""
    if isinstance(content, list):
        # Handle multimodal content lists
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    for keyword, next_agent in ROUTING_MAP.items():
        if keyword in content:
            return next_agent

    # Return None → LLM selector takes over
    return None
