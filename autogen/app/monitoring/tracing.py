import base64
import logging
import os

logger = logging.getLogger(__name__)

_initialized = False
_langfuse_client = None


def init_tracing() -> object | None:
    """
    Initialize Langfuse 4.x + OpenLit tracing via OTLP endpoint.
    Langfuse 4.x is OTel-native — configure OTLP exporter to point at Langfuse Cloud.
    """
    global _initialized, _langfuse_client

    if _initialized:
        return _langfuse_client

    from app.config import settings

    public_key = settings.LANGFUSE_PUBLIC_KEY
    secret_key = settings.LANGFUSE_SECRET_KEY
    host = settings.LANGFUSE_BASE_URL

    if not public_key or not secret_key:
        logger.warning("Langfuse keys not set — tracing disabled.")
        _initialized = True
        return None

    # Set env vars for Langfuse SDK
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key
    os.environ["LANGFUSE_HOST"] = host

    # Configure OTLP exporter → Langfuse OTel endpoint
    # Langfuse 4.x accepts OTel spans at /api/public/otel
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"
    os.environ["OTEL_SERVICE_NAME"] = "content-studio"

    try:
        import openlit
        from langfuse import get_client

        # OpenLit must init FIRST to register the global TracerProvider with
        # the OTLP exporter (pointing at Langfuse). Langfuse SDK then reuses it.
        openlit.init(
            environment=settings.ENVIRONMENT,
            application_name="content-studio",
        )

        langfuse = get_client()

        if not langfuse.auth_check():
            logger.warning("Langfuse auth failed — check keys and LANGFUSE_BASE_URL.")
            _initialized = True
            return None

        _langfuse_client = langfuse
        logger.info("Langfuse tracing initialized (OTLP → %s/api/public/otel)", host)

    except Exception as exc:
        logger.warning("Tracing init failed (%s) — continuing without tracing.", exc)

    _initialized = True
    return _langfuse_client


def get_langfuse():
    """Return the active Langfuse client (may be None if tracing is disabled)."""
    return _langfuse_client
