"""Source 3 of the patient consult: live web search that always returns the pages (URLs) it used.

Search = DuckDuckGo (no key) restricted to reputable medical sites first, plus MedlinePlus' own API (already used by
fallback.py). Matching pages are fetched and reduced to the paragraphs that mention the patient's topic, so the model
answers from real page text and the patient can open the same link.

Privacy: the query is a few anonymised medical keywords (fallback.topic_query); no names, ids, ages or locations leave.
"""
import asyncio
import re
from urllib.parse import urlparse

import httpx

from . import fallback
from .config import get_settings
from .vectorstore import Passage

TRUSTED = {
    "mayoclinic.org": "Mayo Clinic", "nhs.uk": "NHS", "medlineplus.gov": "MedlinePlus (NIH)", "cdc.gov": "CDC",
    "clevelandclinic.org": "Cleveland Clinic", "who.int": "WHO", "hopkinsmedicine.org": "Johns Hopkins Medicine",
    "health.harvard.edu": "Harvard Health", "nih.gov": "NIH", "niddk.nih.gov": "NIDDK", "aad.org": "American Academy of Dermatology",
    "kidshealth.org": "KidsHealth", "healthline.com": "Healthline", "webmd.com": "WebMD", "msdmanuals.com": "MSD Manual",
}
_SITE_FILTER = " OR ".join(f"site:{d}" for d in ("mayoclinic.org", "nhs.uk", "medlineplus.gov", "cdc.gov", "clevelandclinic.org",
                                                  "hopkinsmedicine.org", "who.int", "health.harvard.edu", "msdmanuals.com"))
_BROWSER = {"User-Agent": "Mozilla/5.0 (compatible; healthcare-voice-ai/1.0; research prototype)", "Accept-Language": "en"}


def domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    return next((d for d in TRUSTED if host == d or host.endswith("." + d)), host)


def trusted_name(url: str) -> str | None:
    return TRUSTED.get(domain(url))


def build_query(topic: str) -> str:
    return f"{topic} symptoms causes when to see a doctor"


_BOILERPLATE = re.compile(r"(?:last (?:updated|reviewed|modified)|page was|cookies?|subscribe|newsletter|privacy policy|terms of use|all rights reserved|©|share this|skip to)", re.I)


def relevant_paragraphs(text: str, topic: str, limit_chars: int = 1500) -> str:
    """The paragraphs of a page that share the most words with the topic (page nav/ads rarely do)."""
    terms = {w[:5] for w in re.findall(r"[a-z]{4,}", topic.lower())}
    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 60 and not _BOILERPLATE.search(p[:160])]
    scored = sorted(((len(terms & {w[:5] for w in re.findall(r"[a-z]{4,}", p.lower())}), -i, p) for i, p in enumerate(paras)), reverse=True)
    out, used = [], 0
    for score, _, p in scored:
        if score == 0 and out:
            break
        if used + len(p) > limit_chars:
            p = p[: max(0, limit_chars - used)]
        if p:
            out.append(p)
            used += len(p)
        if used >= limit_chars:
            break
    return " ".join(out)


def extract_text(html: str) -> str:
    try:
        import trafilatura
        got = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
        if got:
            return got
    except Exception:
        pass
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"[ \t]+", " ", re.sub(r"(?s)<[^>]+>", "\n", html))


def _ddg_sync(query: str, n: int) -> list[dict]:
    from ddgs import DDGS
    return list(DDGS().text(query, max_results=n) or [])


async def _fetch_page(client: httpx.AsyncClient, hit: dict, topic: str) -> Passage | None:
    url = hit.get("href") or hit.get("url") or ""
    try:
        r = await client.get(url)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            raise ValueError("not html")
        body = relevant_paragraphs(extract_text(r.text), topic)
    except Exception:
        body = ""
    body = body or (hit.get("body") or "")          # the search engine's own snippet is better than nothing
    if len(body) < 80:
        return None
    name = trusted_name(url)
    return Passage(text=body[:1600], source=name or domain(url), title=(hit.get("title") or domain(url))[:140],
                   score=0.75 if name else 0.45, url=url, direct=False)


async def search(question: str, max_pages: int = 3, *, suffix: str | None = None, names: list[str] | None = None) -> list[Passage]:
    """Passages (each with its URL) from reputable pages about the patient's topic; [] when offline / nothing relevant.

    suffix replaces the default patient-style wording ("symptoms causes when to see a doctor"), e.g. "dosage guideline" for a
    clinician. names are the selected patient's names, removed from the query so they never leave the machine.
    """
    s = get_settings()
    topic = fallback.topic_query(question, names)
    if not s.fallback_enabled or not topic.strip():
        return []
    q = f"{topic} {suffix}" if suffix else build_query(topic)
    hits: list[dict] = []
    for query in (f"{q} ({_SITE_FILTER})", q):         # reputable sites first, then the open web
        try:
            hits = await asyncio.wait_for(asyncio.to_thread(_ddg_sync, query, 8), timeout=15)
        except Exception:
            hits = []
        if any(trusted_name(h.get("href", "")) for h in hits):
            break
    hits.sort(key=lambda h: 0 if trusted_name(h.get("href", "")) else 1)
    seen, picked = set(), []
    for h in hits:
        d = domain(h.get("href", ""))
        if h.get("href") and d not in seen:
            seen.add(d)
            picked.append(h)
        if len(picked) >= max_pages:
            break
    passages: list[Passage] = []
    async with httpx.AsyncClient(timeout=8, headers=_BROWSER, follow_redirects=True) as client:
        pages = await asyncio.gather(*[_fetch_page(client, h, topic) for h in picked], return_exceptions=True)
        passages += [p for p in pages if isinstance(p, Passage)]
        try:                                              # MedlinePlus: authoritative, structured, and has stable URLs
            passages += await fallback._medlineplus(client, topic)
        except Exception:
            pass
    terms = topic.split()
    keep = [p for p in passages if p.url and (trusted_name(p.url) or fallback._relevant(terms, p))]
    keep.sort(key=lambda p: (-(1 if trusted_name(p.url) else 0), -fallback._rank(terms, p)[0], -p.score))
    out, urls = [], set()
    for p in keep:
        if p.url not in urls:
            urls.add(p.url)
            out.append(p)
    return out[:4]
