"""The Clinical Assistant searches the web (trusted medical sites, with links) when its own references have nothing, or when asked."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))
import pytest  # noqa: E402

from app import fallback, llm, rag, vectorstore, websearch  # noqa: E402
from app.vectorstore import Passage  # noqa: E402


def _page():
    return Passage(text="Semaglutide is a GLP-1 receptor agonist used for type 2 diabetes. Guideline updates recommend starting at 0.25 mg weekly.",
                   source="Mayo Clinic", title="Semaglutide", score=0.75, url="https://www.mayoclinic.org/semaglutide", direct=False)


@pytest.fixture
def stubs(monkeypatch):
    calls = {"web": 0}

    async def web(*a, **k):
        calls["web"] += 1
        return [_page()]

    async def none(*a, **k):
        return []
    monkeypatch.setattr(websearch, "search", web)
    monkeypatch.setattr(fallback, "search", none)
    monkeypatch.setattr(vectorstore, "search", lambda *a, **k: [])
    monkeypatch.setattr(llm, "is_available", lambda: asyncio.sleep(0, result=False))
    return calls


def test_web_search_is_used_when_the_references_have_nothing(stubs):
    a = asyncio.run(rag.answer("What is the guideline for semaglutide starting dose?"))
    assert stubs["web"] == 1 and any("mayoclinic.org" in (s.get("url") or "") for s in a.sources)


def test_asking_to_search_the_web_goes_straight_online(stubs):
    a = asyncio.run(rag.answer("Search the web for the latest semaglutide guideline"))
    assert stubs["web"] == 1 and a.sources and a.tier in ("web", "extractive")


def test_patient_specific_questions_never_go_to_the_web(stubs):
    assert rag._WEB_ASK.search("what are the latest guidelines") and not rag._WEB_ASK.search("what is the patient's last HbA1c")


def test_page_boilerplate_is_not_treated_as_an_answer():
    page = ("This page was last updated on this date. Updates may include minor edits to page content and images.\n"
            "Standard drug-sensitive tuberculosis is treated for six months with four medicines in the first two months.")
    out = websearch.relevant_paragraphs(page, "tuberculosis treatment last")
    assert "six months" in out and "last updated" not in out
