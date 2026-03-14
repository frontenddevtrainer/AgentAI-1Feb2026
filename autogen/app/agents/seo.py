from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

SYSTEM_MESSAGE = """
You are an SEO specialist. Review the draft article from WriterAgent and return a
fully SEO-optimized version. Your additions must include:
- Primary keyword placement in H1 and within the first 100 words
- 3-5 secondary keyword variations woven in naturally (no keyword stuffing)
- Optimized meta title (≤60 characters)
- Optimized meta description (≤155 characters)
- 2-3 suggested internal link anchor texts with placeholder URLs

Return the full revised Markdown article followed by an SEO Summary block:

---
**SEO Summary**
- Meta Title: ...
- Meta Description: ...
- Primary Keyword: ...
- Secondary Keywords: ...
- Suggested Internal Links: ...
---

When done, end your message with SEO_COMPLETE.
"""


def make_seo_agent(model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="SEOAgent",
        description="Reviews draft content and injects SEO improvements: keywords, meta tags, internal link anchors.",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
    )
