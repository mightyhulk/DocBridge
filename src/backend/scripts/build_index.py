"""Build the clinical knowledge vector index (ChromaDB).

    python scripts/build_index.py                   # seed KB + any files in data/documents/
    python scripts/build_index.py --patients 20000  # additionally index patient summaries for semantic patient search

Each topic is split into small section-level chunks (one fact group each) so retrieval returns precise passages
and the LLM prompt stays short (lower latency).
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import db, vectorstore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embeddings import embed_documents  # noqa: E402
from kb_seed import CONDITIONS, DRUGS, REFERENCE  # noqa: E402


def seed_chunks() -> list[tuple[str, str, dict]]:
    out = []
    for name, c in CONDITIONS.items():
        base = {"title": name, "source": "Clinical reference (seed)"}
        out.append((f"{name}: {c['overview']}", f"cond::{name}::overview", base))
        out.append((f"{name} symptoms: " + "; ".join(c["symptoms"]) + ".", f"cond::{name}::symptoms", base))
        out.append((f"{name} red flags - when to seek urgent care: " + "; ".join(c["red_flags"]) + ".", f"cond::{name}::red", base))
        out.append((f"{name} management and treatment: " + "; ".join(c["management"]) + ". Medicines: " + "; ".join(c["medications"]) + ".",
                    f"cond::{name}::mgmt", base))
    for name, d in DRUGS.items():
        base = {"title": name, "source": "Drug reference (seed)"}
        aka = f" (also called {d['also']})" if d["also"] else ""
        out.append((f"{name}{aka} is a {d['class']} used for {d['uses']}.", f"drug::{name}::about", base))
        out.append((f"{name} dose and dosing: {d['dose']}", f"drug::{name}::dose", base))
        out.append((f"{name} side effects: {d['side_effects']} Warnings and contraindications: {d['warnings']}", f"drug::{name}::safety", base))
    for title, text in REFERENCE.items():
        out.append((f"{title}: {text}", f"ref::{title}", {"title": title, "source": "Clinical reference (seed)"}))
    return out


def document_chunks(folder: Path) -> list[tuple[str, str, dict]]:
    """User-supplied .md/.txt documents: split on blank lines into ~600 char chunks with a little overlap."""
    out = []
    for f in sorted(folder.glob("**/*")):
        if f.suffix.lower() not in (".md", ".txt"):
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", f.read_text(encoding="utf-8", errors="ignore")) if p.strip()]
        buf, n = "", 0
        for p in paras + [""]:
            if p and len(buf) + len(p) < 600:
                buf += ("\n" if buf else "") + p
                continue
            if buf:
                out.append((buf, f"doc::{f.name}::{n}", {"title": f.stem.replace("_", " "), "source": f"Document: {f.name}"}))
                n += 1
            buf = p
    return out


def upsert(col, chunks, batch=128) -> None:
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.upsert(ids=[c[1] for c in part], documents=[c[0] for c in part], metadatas=[c[2] for c in part],
                   embeddings=embed_documents([c[0] for c in part]))
        print(f"  indexed {min(i + batch, len(chunks)):,}/{len(chunks):,}", end="\r")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=0, help="also embed N patient summaries (slow on CPU)")
    args = ap.parse_args()
    s = get_settings()
    docs_dir = s.data_dir / "documents"
    docs_dir.mkdir(exist_ok=True)
    col = vectorstore.collection()
    t0 = time.time()
    chunks = seed_chunks() + document_chunks(docs_dir)
    print(f"Embedding {len(chunks)} knowledge chunks with {s.embed_model} ...")
    upsert(col, chunks)
    if args.patients:
        rows = db.conn().execute(
            "SELECT patient_id, full_name, age, sex, conditions, medications, allergies FROM patients LIMIT ?", (args.patients,)).fetchall()
        pc = [(f"Patient {r['patient_id']} {r['full_name']}, {r['age']}y {r['sex']}. Conditions: {r['conditions'] or 'none'}. "
               f"Medications: {r['medications'] or 'none'}. Allergies: {r['allergies'] or 'none'}.",
               f"pat::{r['patient_id']}", {"title": r["patient_id"], "source": "EHR summary"}) for r in rows]
        print(f"Embedding {len(pc):,} patient summaries ...")
        upsert(col, pc, batch=256)
    print(f"Done in {time.time() - t0:.0f}s. Vector DB now holds {col.count():,} documents at {s.chroma_dir}")


if __name__ == "__main__":
    main()
