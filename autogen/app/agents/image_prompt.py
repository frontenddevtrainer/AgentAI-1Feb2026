from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_MESSAGE = """
You are a visual content director. Based on the article content, produce image
generation prompts suitable for DALL-E or Midjourney:

1. **Hero Image** (16:9 aspect ratio, max 100 words)
   - Specify style (photorealistic / illustrated / infographic)
   - Include color palette suggestion
   - Include mood descriptor

2. **Section Illustration 1** (square format, max 60 words)
   - Should relate to the article's first or second main section

3. **Section Illustration 2** (square format, max 60 words)
   - Should relate to a different section from illustration 1

Format each prompt clearly labeled. Make prompts vivid and specific enough to
generate consistent, on-brand visuals.
When done, end your message with IMAGES_COMPLETE.
"""


def make_image_prompt_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="ImagePromptAgent",
        description="Creates detailed image generation prompts for hero images and section illustrations.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
    )
