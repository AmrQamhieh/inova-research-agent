import os


# Provide minimal environment configuration for module imports during tests.
# Individual tests can override these via monkeypatch if needed.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_MODEL", "test-model")

