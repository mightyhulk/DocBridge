"""Build the patient-doctor dialogue index used by the patient consult (Source 2).

    python scripts/build_dialogue_index.py                 # both datasets (~35 min on a laptop CPU; resumable)
    python scripts/build_dialogue_index.py --limit 5000    # quick trial

Downloads the parquet files from Hugging Face on first run, cleans them (greetings/sign-offs/adverts removed, duplicates
dropped) and embeds each row by the patient's description. Safe to stop and re-run: finished rows are skipped.
Stop the backend first if it is running (it must reload the index anyway).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from app import dialogues as D  # noqa: E402
from app.config import get_settings  # noqa: E402

DATA = get_settings().data_dir / "datasets"


def rows_chatdoctor():
    p = hf_hub_download("lavita/ChatDoctor-HealthCareMagic-100k", "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet",
                        repo_type="dataset", local_dir=str(DATA / "chatdoctor"))
    for i, r in enumerate(pq.read_table(p).to_pylist()):
        yield "cd", i, "", r["input"], r["output"]


def rows_amc():
    p = hf_hub_download("ruslanmv/ai-medical-chatbot", "dialogues.parquet", repo_type="dataset",
                        local_dir=str(DATA / "ai-medical-chatbot"))
    for i, r in enumerate(pq.read_table(p).to_pylist()):
        title = str(r.get("Description") or "").removeprefix("Q.").strip()
        yield "amc", i, title, r["Patient"], r["Doctor"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max rows per dataset (0 = all)")
    ap.add_argument("--only", choices=["cd", "amc"], help="index just one dataset")
    args = ap.parse_args()

    col = D.collection()
    done = set()
    seen_replies: set[str] = set()
    total = col.count()
    for off in range(0, total, 2000):                      # page: Chroma cannot return 50k+ rows in one call
        got = col.get(limit=2000, offset=off, include=["metadatas"])
        done.update(got["ids"])
        seen_replies.update(D.reply_hash(m["reply"]) for m in got["metadatas"])
    print(f"index has {len(done)} rows already")

    t0 = time.time()
    for gen in ([rows_chatdoctor, rows_amc] if not args.only else [rows_chatdoctor if args.only == "cd" else rows_amc]):
        batch_ids, batch_docs, batch_meta, batch_emb_text = [], [], [], []
        kept = skipped = 0

        def flush():
            nonlocal batch_ids, batch_docs, batch_meta, batch_emb_text
            if not batch_ids:
                return
            col.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=D.embed(batch_emb_text))
            batch_ids, batch_docs, batch_meta, batch_emb_text = [], [], [], []

        for n, (prefix, i, title, patient, doctor) in enumerate(gen()):
            if args.limit and n >= args.limit:
                break
            rid = D.row_id(prefix, i)
            if rid in done:
                continue
            patient, reply = D.clean_patient(patient), D.clean_doctor(doctor)
            h = D.reply_hash(reply)
            if not D.usable_row(patient, reply) or h in seen_replies:
                skipped += 1
                continue
            seen_replies.add(h)
            name, url = D.DATASETS[prefix]
            title = (title or patient.split(".")[0])[:140]
            batch_ids.append(rid)
            batch_docs.append(patient[:700])
            batch_meta.append({"title": title, "reply": reply[:1800], "source": name, "url": url})
            batch_emb_text.append(D.embed_text_for(patient, title))
            kept += 1
            if len(batch_ids) >= 256:
                flush()
                if kept % 2560 == 0:
                    print(f"  {prefix}: kept {kept}, skipped {skipped}, total index {col.count()}  ({time.time() - t0:.0f}s)", flush=True)
        flush()
        print(f"{gen.__name__}: kept {kept}, skipped {skipped}", flush=True)
    print(f"done: {col.count()} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
