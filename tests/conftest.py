import pytest


@pytest.fixture(autouse=True)
def _no_live_web_search(monkeypatch):
    """Unit tests never reach the real internet through the Clinical Assistant's web tier; tests that need it stub their own."""
    from app import websearch

    async def offline(*a, **k):
        return []
    monkeypatch.setattr(websearch, "search", offline)
