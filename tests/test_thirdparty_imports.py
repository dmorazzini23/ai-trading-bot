# AI-AGENT-REF: ensure core third-party libraries available

def test_import_core_thirdparty():
    import pandas  # noqa: F401
    import pydantic  # noqa: F401
