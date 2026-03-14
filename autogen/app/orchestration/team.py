from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

from app.agents.base import get_model_client
from app.agents.ideation import make_ideation_agent
from app.agents.outline import make_outline_agent
from app.agents.writer import make_writer_agent
from app.agents.seo import make_seo_agent
from app.agents.image_prompt import make_image_prompt_agent
from app.agents.publisher import make_publisher_agent
from app.orchestration.selector import SELECTOR_PROMPT, selector_func


def build_team() -> SelectorGroupChat:
    """
    Assembles all 6 content agents into a SelectorGroupChat.
    Creates a fresh team per request — correct isolation for multi-user Gradio.
    """
    model_client = get_model_client()

    agents = [
        make_ideation_agent(model_client),
        make_outline_agent(model_client),
        make_writer_agent(model_client),
        make_seo_agent(model_client),
        make_image_prompt_agent(model_client),
        make_publisher_agent(model_client),
    ]

    termination = (
        TextMentionTermination("PUBLISH_READY")
        | TextMentionTermination("TERMINATE")
        | MaxMessageTermination(max_messages=30)
    )

    return SelectorGroupChat(
        participants=agents,
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        selector_func=selector_func,
        termination_condition=termination,
        allow_repeated_speaker=False,
    )


if __name__ == "__main__":
    import asyncio

    async def _test():
        team = build_team()
        async for event in team.run_stream(
            task="Write a blog post about async Python for beginners"
        ):
            source = getattr(event, "source", type(event).__name__)
            content = str(getattr(event, "content", ""))[:120]
            print(f"[{source}] {content}")

    asyncio.run(_test())
