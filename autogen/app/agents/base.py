from autogen_ext.models.openai import OpenAIChatCompletionClient
from app.config import settings


def get_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
