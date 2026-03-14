from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_MESSAGE = """
You are a content publisher and final editor. Perform a comprehensive quality check
and package all outputs into a final deliverable.

Steps:
1. Confirm the article reads coherently from start to finish
2. Verify all SEO elements are present (meta title, meta description, keywords)
3. Verify image prompts are present and labeled correctly
4. Estimate reading time (words / 200 = minutes)

Produce a final JSON package with EXACTLY these keys:
{
  "title": "...",
  "meta_title": "...",
  "meta_description": "...",
  "body_markdown": "...",
  "image_prompts": {
    "hero": "...",
    "section_1": "...",
    "section_2": "..."
  },
  "seo": {
    "primary_keyword": "...",
    "secondary_keywords": [],
    "internal_link_anchors": []
  },
  "tags": [],
  "estimated_read_time_minutes": 0,
  "warnings": []
}

Output ONLY the JSON block, no extra text before or after.
When done, end your message with PUBLISH_READY.
"""


def make_publisher_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="PublisherAgent",
        description="Performs final quality check, formats content for publishing, and produces the final deliverable JSON.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
    )
