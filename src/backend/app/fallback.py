"""Tier 3: free online sources (no API keys): MedlinePlus, openFDA drug labels, DuckDuckGo instant answers
(Wikipedia-backed) and, as a last resort, PubMed abstracts.

Only an anonymised topic query (a few medical keywords) ever leaves the machine; see topic_query().
"""
import asyncio
import re
import xml.etree.ElementTree as ET

import httpx

from . import grounding, safety
from .config import get_settings
from .vectorstore import Passage

_UA = {"User-Agent": "healthcare-voice-ai/1.0 (research prototype)"}

# Question-shape words that are useless (or harmful) in a search query.
_FILLER = set("high low level levels mean means meaning work works working difference differences between warning "
              "caused form forms treated diagnosed diagnose diagnosis tests test normal range safe safely".split())
# Lay phrasing -> the term the reference sources use.
_LAY = {"colon cancer": "colorectal cancer", "stomach flu": "gastroenteritis", "high blood pressure": "hypertension",
        "sugar disease": "diabetes", "heart attack": "myocardial infarction"}

# openFDA label sections by what the question is about: (field, heading)
_LABEL_INTENTS = [
    (r"side effect|adverse|reaction", [("adverse_reactions", "Adverse reactions")]),
    (r"interact|together with|combine|mix|take .* with", [("drug_interactions", "Interactions")]),
    (r"pregnan|breast|lactat|nursing|fertility", [("pregnancy", "Pregnancy"), ("nursing_mothers", "Nursing mothers"),
                                                   ("use_in_specific_populations", "Use in specific populations")]),
    (r"\bdos(e|age|ing)\b|how much|how often|\bmg\b", [("dosage_and_administration", "Dosage")]),
    (r"mechanism|how does .* work|how do .* work|works?\b", [("mechanism_of_action", "Mechanism of action"),
                                                            ("clinical_pharmacology", "Clinical pharmacology")]),
    (r"contraindic|who should not|avoid", [("contraindications", "Contraindications")]),
    (r"warning|risk|danger|boxed", [("boxed_warning", "Boxed warning"), ("warnings", "Warnings")]),
]
_DEFAULT_LABEL = [("indications_and_usage", "Indications"), ("warnings", "Warnings")]


def topic_query(question: str, names: list[str] | None = None) -> str:
    """Reduce a question to anonymous topic keywords: 'What causes tension headaches?' -> 'tension headaches'.

    Strips IDs/phones/emails/dates, the selected patient's name and possessive names ("Rahul's"), then question-shape
    words. Keeps digits and single letters that carry meaning ('type 1', 'hepatitis B').
    """
    text = re.sub(r"\b[A-Z][a-z]+'s\b", " ", safety.mask_pii(question, names))
    out = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+-]*", text):
        lw = w.lower()
        if lw in safety._STOP or lw in grounding._GENERIC or lw in _FILLER:
            continue
        if w in ("PATIENT_ID", "EMAIL", "PHONE", "DATE", "NUMBER", "NAME"):
            continue
        out.append(w)
    q = " ".join(out[:8])
    for lay, term in _LAY.items():
        q = re.sub(lay, term, q, flags=re.I)
    for abbr, full in _ABBR.items():            # lab/test abbreviations search poorly ('CRP' matches 'Complex Regional Pain')
        q = re.sub(rf"\b{abbr}\b", full, q, flags=re.I)
    return q


_ABBR = {"crp": "C-reactive protein", "esr": "erythrocyte sedimentation rate", "tsh": "thyroid stimulating hormone",
         "bnp": "brain natriuretic peptide", "inr": "INR blood test", "cbc": "complete blood count", "gfr": "glomerular filtration rate",
         "egfr": "glomerular filtration rate", "psa": "prostate specific antigen", "ecg": "electrocardiogram", "ekg": "electrocardiogram",
         "ldl": "LDL cholesterol", "hdl": "HDL cholesterol", "alt": "ALT blood test", "ast": "AST blood test"}

_COMPARE = re.compile(r"(?:difference|differences|compare|comparison|vs\.?|versus)\s+(?:between\s+)?(.+?)\s+(?:and|vs\.?|versus|from|with)\s+(.+?)\s*[?.]?$", re.I)


def comparison_topics(question: str, names: list[str] | None = None) -> list[str]:
    """'difference between type 1 and type 2 diabetes' -> ['type 1 diabetes', 'type 2 diabetes'] (searched separately)."""
    m = _COMPARE.search(question)
    if not m:
        return []
    a, b = (topic_query(x, names) for x in m.groups())
    if not a or not b:
        return []
    ta, tb = a.split(), b.split()
    if len(tb) >= 2 and tb[-1].lower() not in (w.lower() for w in ta):      # shared head noun: 'type 1' + 'diabetes'
        ta.append(tb[-1])
    return [" ".join(ta), " ".join(tb)]


def label_fields(question: str) -> list[tuple[str, str]]:
    q = question.lower()
    for pat, fields in _LABEL_INTENTS:
        if re.search(pat, q):
            return fields
    return _DEFAULT_LABEL


def _clean(html: str, limit: int = 900) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


async def _medlineplus(client: httpx.AsyncClient, term: str) -> list[Passage]:
    r = await client.get("https://wsearch.nlm.nih.gov/ws/query",
                         params={"db": "healthTopics", "term": term, "retmax": 3, "rettype": "brief"})
    r.raise_for_status()
    out = []
    for doc in ET.fromstring(r.text).iter("document"):
        fields = {c.get("name"): "".join(c.itertext()) for c in doc.findall("content")}
        title = _clean(fields.get("title", ""), 120)
        summary = _clean(fields.get("FullSummary", "") or fields.get("snippet", ""), 1500)
        if summary:
            out.append(Passage(text=summary, source="MedlinePlus (NIH)", title=title, score=0.7, url=doc.get("url", "")))
    return out[:3]


async def _label(client: httpx.AsyncClient, drug: str, fields: list[tuple[str, str]]) -> list[Passage]:
    """Official FDA label sections for one drug. Prefers oral products (an eye-drop label once answered a dosing question)."""
    # A real drug has its own name as the active substance. (Homeopathic products are *named* after conditions, e.g.
    # a product called "Gout", so a brand-name match is accepted only for prescription drugs.)
    ident = (f'(openfda.substance_name:"{drug}" OR (openfda.brand_name:"{drug}" '
             f'AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"))')
    for route_filter in (' AND openfda.route:"ORAL"', ""):
        r = await client.get("https://api.fda.gov/drug/label.json", params={"search": ident + route_filter, "limit": 8})
        if r.status_code != 200:
            continue
        labels = r.json().get("results", [])
        for wanted in (fields, _DEFAULT_LABEL):
            for label in labels:
                out = [Passage(text=_clean(" ".join(label[f]) if isinstance(label[f], list) else str(label[f]), 1500),
                               source="openFDA drug label", title=f"{drug.title()} - {heading}", score=0.8,
                               url="https://open.fda.gov/apis/drug/label/", direct=True)
                       for f, heading in wanted if label.get(f)]
                if out:
                    return out[:2]
    return []


async def _duckduckgo(client: httpx.AsyncClient, term: str) -> list[Passage]:
    r = await client.get("https://api.duckduckgo.com/", params={"q": term, "format": "json", "no_html": 1, "skip_disambig": 1})
    r.raise_for_status()
    d = r.json()
    text = d.get("AbstractText", "")
    if not text:
        return []
    return [Passage(text=text[:1500], source=f"{d.get('AbstractSource') or 'Wikipedia'} (via DuckDuckGo)",
                    title=d.get("Heading", term), score=0.6, url=d.get("AbstractURL", ""))]


async def _pubmed(client: httpx.AsyncClient, term: str) -> list[Passage]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    r = await client.get(f"{base}/esearch.fcgi", params={"db": "pubmed", "term": f"{term} AND hasabstract AND (review[pt] OR guideline[pt])",
                                                         "retmax": 2, "retmode": "json", "sort": "relevance"})
    r.raise_for_status()
    ids = r.json()["esearchresult"]["idlist"]
    if not ids:
        return []
    r = await client.get(f"{base}/efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
    r.raise_for_status()
    out = []
    for art in ET.fromstring(r.text).iter("PubmedArticle"):   # (relevance is enforced afterwards: the title must name the topic)
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
        pmid = art.findtext(".//PMID", "")
        if abstract:
            out.append(Passage(text=abstract[:1500], source="PubMed (review abstract)", title=title[:140], score=0.5,
                               url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
    return out


def _relevant(topic_terms: list[str], p: Passage) -> bool:
    """A page is kept only if it is about the topic: a topic word is in its title or opening sentence."""
    if p.direct:
        return True
    stems = {t.lower()[:5] for t in topic_terms}
    if p.source.startswith("PubMed"):      # research abstracts are only usable when the title itself names the topic
        return bool(stems & grounding._stems(p.title))
    return bool(stems & grounding._stems(p.title + " " + p.text[:160]))


def _rank(topic_terms: list[str], p: Passage) -> tuple[int, float]:
    stems = {t.lower()[:5] for t in topic_terms}
    return (len(stems & grounding._stems(p.title)), p.score)


async def search(question: str, names: list[str] | None = None) -> list[Passage]:
    """Query the free sources concurrently; individual failures never break the pipeline."""
    s = get_settings()
    topic = topic_query(question, names)
    if not s.fallback_enabled or not topic.strip():
        return []
    terms = topic.split()
    drugs = [t.lower() for t in terms if not grounding.is_common_word(t.lower()) and t.isalpha()][:3]
    fields = label_fields(question)
    topics = comparison_topics(question, names) or [topic]
    if len(topics) > 1:
        terms = [w for t in topics for w in t.split()]
    async with httpx.AsyncClient(timeout=s.fallback_timeout_s, headers=_UA, follow_redirects=True) as client:
        jobs = []
        for tp in topics:
            phrase = [f'"{tp}"', tp] if len(tp.split()) > 1 else [tp]
            jobs += [_medlineplus(client, v) for v in phrase] + [_duckduckgo(client, tp)]
        jobs += [_label(client, d, fields) for d in drugs]
        results = await asyncio.gather(*jobs, return_exceptions=True)
        passages = _merge(terms, results)
        if not passages:   # last resort: published review abstracts
            try:
                passages = _merge(terms, [await _pubmed(client, topic)])
            except Exception:
                passages = []
    return passages


def _merge(terms: list[str], results: list) -> list[Passage]:
    seen, passages = set(), []
    for r in results:
        for p in (r if isinstance(r, list) else []):
            if p.text and (p.source, p.title) not in seen and _relevant(terms, p):
                seen.add((p.source, p.title))
                passages.append(p)
    passages.sort(key=lambda p: _rank(terms, p), reverse=True)
    return passages[:4]
