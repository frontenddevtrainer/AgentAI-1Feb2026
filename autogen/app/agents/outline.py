from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_MESSAGE = """
You are a content architect. Given a content idea from IdeationAgent, produce a
full article outline with:
- H1 title
- Meta description draft (1-2 sentences)
- 4-6 H2 sections, each with 2-3 bullet point sub-topics
- Estimated word count per section
- A call-to-action (CTA) suggestion at the end

Output in valid Markdown. Be logical in flow — intro to conclusion.
When done, end your message with OUTLINE_COMPLETE.
"""


def make_outline_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="OutlineAgent",
        description="Converts a chosen content idea into a detailed, structured article outline.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
    )
