from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_MESSAGE = """
You are a creative content strategist. Given a topic or brief, generate 3-5 distinct
content angles. For each angle provide:
- A working title
- Target audience
- Core hook (one compelling sentence)
- 2-sentence rationale explaining why this angle works

Output as a structured Markdown list. Be specific, original, and audience-focused.
When done, end your message with IDEATION_COMPLETE.
"""


def make_ideation_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="IdeationAgent",
        description="Generates creative content ideas, angles, and topic variations from a user brief.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
    )
