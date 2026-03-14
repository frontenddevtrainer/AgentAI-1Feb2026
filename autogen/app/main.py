import logging

from app.config import settings
from app.monitoring.tracing import init_tracing
from app.ui.components import create_ui, THEME, CSS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting AI Content Creation Studio...")
    init_tracing()

    demo = create_ui()
    demo.launch(
        server_name=settings.GRADIO_SERVER_NAME,
        server_port=settings.GRADIO_SERVER_PORT,
        debug=(settings.ENVIRONMENT == "development"),
        show_error=True,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
