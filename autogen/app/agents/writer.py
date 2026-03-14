from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_MESSAGE = """
You are an expert long-form content writer. Using the outline provided by OutlineAgent,
write the complete article. Follow these guidelines:
- Match the structure from the outline (H1, H2s, sub-topics)
- Maintain consistent, engaging voice throughout
- Use natural transitions between sections
- Include relevant examples, analogies, or data points
- Target 800-1500 words unless the outline specifies otherwise
- End with the CTA suggested in the outline

Output the full article in Markdown format.
When done, end your message with DRAFT_COMPLETE.
"""


def make_writer_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="WriterAgent",
        description="Writes the full article body from the outline, producing long-form draft content.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
    )
