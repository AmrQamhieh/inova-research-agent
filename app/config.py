import os


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv_if_enabled() -> None:
    """
    12-factor: configuration comes from the environment.

    For local development, you may opt-in to loading a `.env` file by setting
    `LOAD_DOTENV=true`. In all other environments, nothing is loaded implicitly.
    """

    if not _env_flag("LOAD_DOTENV"):
        return

    # Imported lazily so production doesn't depend on python-dotenv.
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(override=False)

