# Add every supported file in documents/ to the Ext vector store (ChromaDB, chroma_db_global/).
#
# CLI usage:
#   python -m ext_rag.ingest                 add new files and re-embed changed ones
#   python -m ext_rag.ingest --remove FILE   delete one document from the store
#   python -m ext_rag.ingest --rebuild       delete the store and embed everything again
#
# UI helpers (used by other_rags/registry.py for unified Streamlit sidebar management):
#   index_ext_file(pdf_source)
#   get_ext_indexed_files()
#   remove_ext_file(file_name)

import os
import sys
import re
import csv
import time
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag import cost
from hybrid_rag.logging_setup import setup_logging
from hybrid_rag.pdf_processing import get_valid_pdf_path, resolve_file_name
from ext_rag.rag_common import (
    RAGError,
    DOCS_DIR,
    STORE_DIR,
    COLLECTION,
    CHUNK_CHARS,
    OVERLAP_CHARS,
    BACKEND,
    BATCH,
    EMBED_MODEL,
    embed_texts,
    open_store,
    store_exists,
    delete_store,
    stored_files,
    check_dims,
    fail,
)

logger = logging.getLogger("rag.ingest")


# ---------- Loaders: each returns (kind, [(label, text), ...]) ----------
def load_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "page", [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]


def load_docx(path):
    from docx import Document
    doc = Document(path)
    paras = [p for p in doc.paragraphs if p.text.strip()]

    def is_heading(p):
        name = (p.style.name if p.style is not None else "").lower()
        return name.startswith("heading") or name == "title"

    units = []
    if any(is_heading(p) for p in paras):
        kind, section = "section", "Beginning"
        for p in paras:
            if is_heading(p):
                section = p.text.strip()[:60]
            units.append((section, p.text))
    else:
        kind = "paragraph"
        units = [(i, p.text) for i, p in enumerate(paras, 1)]

    for t, table in enumerate(doc.tables, 1):
        label = f"Table {t}" if kind == "section" else len(units) + 1
        for row in table.rows:
            cells = []
            for c in row.cells:
                txt = c.text.strip()
                if txt and (not cells or cells[-1] != txt):
                    cells.append(txt)
            if cells:
                units.append((label, " | ".join(cells)))
    return kind, units


def load_pptx(path):
    try:
        from pptx import Presentation
    except ImportError:
        fail("python-pptx is required to ingest .pptx files. Install it with: uv pip install python-pptx")
    units = []
    for i, slide in enumerate(Presentation(path).slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text for c in row.cells))
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
        units.append((i, "\n".join(parts)))
    return "slide", units


def read_text(path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def load_txt(path):
    return "line", list(enumerate(read_text(path).splitlines(), 1))


def load_csv(path):
    rows = list(csv.reader(read_text(path).splitlines()))
    if not rows:
        return "row", []
    header, units = rows[0], []
    for i, row in enumerate(rows[1:], 1):
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        units.append((i, "; ".join(pairs) + "."))
    return "row", units


def load_html(path):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        fail("beautifulsoup4 is required to ingest .html files. Install it with: uv pip install beautifulsoup4")
    soup = BeautifulSoup(read_text(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    return "line", list(enumerate(lines, 1))


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".txt": load_txt,
    ".md": load_txt,
    ".csv": load_csv,
    ".html": load_html,
    ".htm": load_html,
}


# ---------- Chunking ----------
def format_location(kind, labels):
    labels = list(dict.fromkeys(labels))
    if all(isinstance(x, int) for x in labels):
        lo, hi = min(labels), max(labels)
        return f"{kind} {lo}" if lo == hi else f"{kind}s {lo}-{hi}"
    shown = "; ".join(str(x) for x in labels[:3])
    return f"{kind}: {shown}" + (" ..." if len(labels) > 3 else "")


SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_CHARS,
    chunk_overlap=OVERLAP_CHARS,
    separators=["\n\n", "\n", " ", ""],
)


def make_blocks(kind, units):
    if kind in ("page", "slide"):
        return [([label], text) for label, text in units]
    blocks, labels, parts, size = [], [], [], 0
    for label, text in units:
        labels.append(label)
        parts.append(text)
        size += len(text)
        if size >= 4 * CHUNK_CHARS:
            blocks.append((labels, "\n".join(parts)))
            labels, parts, size = [], [], 0
    if parts:
        blocks.append((labels, "\n".join(parts)))
    return blocks


def make_chunks(kind, units):
    chunks = []
    for labels, text in make_blocks(kind, units):
        for piece in SPLITTER.split_text(text):
            cleaned = re.sub(r"\s+", " ", piece).strip()
            if len(cleaned) > 30:
                chunks.append((format_location(kind, labels), cleaned))
    n_words = sum(len(text.split()) for _, text in units)
    logger.debug("%d words in %d %ss -> %d chunks of up to %d characters (%d overlap)",
                 n_words, len(units), kind, len(chunks), CHUNK_CHARS, OVERLAP_CHARS)
    return chunks, n_words


# ---------- Store maintenance ----------
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_file(col, digest):
    col.delete(where={"hash": digest})
    logger.debug("Deleted the chunks of %s from the store", digest[:12])


def stored_chunks(col, digest):
    return {m["chunk"] for m in col.get(where={"hash": digest}, include=["metadatas"])["metadatas"]}


def add_file(col, path, rel, digest, done=frozenset()):
    loader = LOADERS[path.suffix.lower()]
    started = time.perf_counter()
    kind, units = loader(path)
    logger.debug("%s read %s: %d %ss in %.1fs", loader.__name__, rel, len(units), kind,
                 time.perf_counter() - started)
    chunks, n_words = make_chunks(kind, units)
    if n_words < 20:
        logger.warning("Skipped %s: only %d words found", rel, n_words)
        return 0

    todo = [i for i in range(len(chunks)) if i not in done]
    logger.info("Ext ingest '%s': %d words -> %d chunks (%d already saved, embedding %d)",
                rel, n_words, len(chunks), len(done), len(todo))

    added = datetime.now().isoformat(timespec="seconds")
    rejected = []
    for start in range(0, len(todo), BATCH):
        part = todo[start:start + BATCH]
        vectors = embed_texts([chunks[i][1] for i in part], skip_rejected=True)
        keep = [(i, v) for i, v in zip(part, vectors) if v is not None]
        rejected += [i for i, v in zip(part, vectors) if v is None]
        if keep:
            check_dims(col, len(keep[0][1]), record=True)
            col.upsert(
                ids=[f"{digest}:{i}" for i, _ in keep],
                embeddings=np.vstack([v for _, v in keep]),
                documents=[chunks[i][1] for i, _ in keep],
                metadatas=[{"path": rel, "location": chunks[i][0], "hash": digest,
                            "file_type": path.suffix.lower().lstrip("."), "added": added,
                            "chunk": i, "chunks": len(chunks)}
                           for i, _ in keep])
        saved = len(done) + start + len(part)
        logger.debug("Saved %d chunks of %s, %d of %d done", len(keep), rel, saved, len(chunks))

    if rejected:
        ids = col.get(where={"hash": digest}, include=[])["ids"]
        if ids:
            col.update(ids=ids, metadatas=[{"skipped": len(rejected)}] * len(ids))
        logger.warning("Skipped %d chunks of %s refused by the content safety screen: %s",
                       len(rejected), rel, ", ".join(f"chunk {i} ({chunks[i][0]})" for i in rejected))
        if not ids:
            return 0
    logger.info("Embedded %s (%s): %d words -> %d chunks, %d skipped", rel, digest[:12],
                n_words, len(chunks), len(rejected))
    return len(chunks)


# ---------- Unified UI / Registry Helpers ----------
def index_ext_file(pdf_source):
    """
    Indexes a PDF file (file path or Streamlit UploadedFile) into ext_chunks_<backend>.
    Skips if the file (same content hash or already complete by filename) is present.
    """
    if isinstance(pdf_source, Path):
        pdf_source = str(pdf_source)
    file_name = resolve_file_name(pdf_source)
    is_temp = not isinstance(pdf_source, str) and not (
        hasattr(pdf_source, "name") and isinstance(pdf_source.name, str) and os.path.exists(pdf_source.name)
    )
    valid_path = Path(get_valid_pdf_path(pdf_source))
    try:
        col = open_store()
        stored = stored_files(col)
        digest = file_hash(valid_path)

        if digest in stored and stored[digest]["complete"]:
            logger.info("Document '%s' is already indexed in %s. Skipping.", file_name, COLLECTION)
            return

        by_path = {f["path"]: h for h, f in stored.items() if f["complete"]}
        old = by_path.get(file_name)
        done = stored_chunks(col, digest) if digest in stored else set()

        if add_file(col, valid_path, file_name, digest, done):
            if old and old != digest:
                remove_file(col, old)
    finally:
        if is_temp:
            valid_path.unlink(missing_ok=True)


def get_ext_indexed_files() -> list:
    """Returns a sorted list of unique filenames stored in ext_chunks_<backend>."""
    if not store_exists():
        return []
    col = open_store(check=False)
    if col.count() == 0:
        return []
    metas = col.get(include=["metadatas"]).get("metadatas") or []
    paths = {m.get("path") for m in metas if isinstance(m, dict) and m.get("path")}
    return sorted(paths)


def remove_ext_file(file_name: str):
    """Removes all chunks belonging to file_name from ext_chunks_<backend>."""
    if not store_exists():
        return
    col = open_store(check=False)
    if col.count() == 0:
        return
    logger.info("Removing '%s' from %s...", file_name, COLLECTION)
    col.delete(where={"path": file_name})
    logger.info("Successfully removed '%s' from %s.", file_name, COLLECTION)


# ---------- CLI ----------
def run():
    parser = argparse.ArgumentParser(description="Add documents to the Ext vector store")
    parser.add_argument("--rebuild", action="store_true", help="start from an empty store")
    parser.add_argument("--force", action="store_true",
                        help="with --rebuild: also lose documents that are no longer in the folder")
    parser.add_argument("--remove", metavar="FILE",
                        help="delete one document from the store (path as shown by ext_rag.ask --list)")
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    on_disk = {}
    for p in sorted(DOCS_DIR.rglob("*")):
        if not p.is_file() or p.name.startswith(("~$", ".")):
            continue
        if p.suffix.lower() in LOADERS:
            on_disk[p.relative_to(DOCS_DIR).as_posix()] = p
        else:
            print(f"Unsupported file type, ignored: {p.name}")
    logger.info("Found %d supported files in %s", len(on_disk), DOCS_DIR)

    if args.rebuild and store_exists():
        only_in_store = sorted({f["path"] for f in stored_files(open_store(check=False)).values()}
                               - set(on_disk))
        if only_in_store and not args.force:
            fail("These documents are in the store but no longer in documents/, so a rebuild "
                 "would lose them:\n  " + "\n  ".join(only_in_store) +
                 "\nPut them back in the folder, or add --force to rebuild without them.")
        delete_store()
        print(f"Deleted the old {COLLECTION} store")

    print(f"Embedding model: {EMBED_MODEL} ({'local, Ollama' if BACKEND == 'ollama' else 'online, OpenRouter'}) -> {STORE_DIR.name}/{COLLECTION}")
    col = open_store()
    stored = stored_files(col)

    if args.remove:
        hashes = [h for h, f in stored.items() if f["path"] == args.remove]
        if not hashes:
            fail(f'"{args.remove}" is not in the store. See the list with: python -m ext_rag.ask --list')
        for h in hashes:
            remove_file(col, h)
        print(f"Removed from the store: {args.remove}")
        logger.info("Removed %s on request", args.remove)
        return

    digests = {rel: file_hash(path) for rel, path in on_disk.items()}

    for h, f in list(stored.items()):
        if not f["complete"] and h not in digests.values():
            remove_file(col, h)
            del stored[h]
            logger.warning("Removed the unfinished import of %s (%d of %d chunks): the file "
                           "is no longer in the folder", f["path"], f["stored"], f["chunks"])

    by_path = {f["path"]: h for h, f in stored.items() if f["complete"]}
    added = updated = unchanged = failed = 0
    for rel, path in on_disk.items():
        digest = digests[rel]
        if digest in stored and stored[digest]["complete"]:
            unchanged += 1
            if stored[digest]["path"] != rel:
                print(f"Already in the store as {stored[digest]['path']}, not embedded again: {rel}")
            continue
        old = by_path.get(rel)
        done = stored_chunks(col, digest) if digest in stored else set()
        print(f"{'Continuing' if done else 'Updating' if old else 'Adding'}: {rel}")
        qc = cost.start_query("ingest", rel)
        try:
            if add_file(col, path, rel, digest, done):
                stored[digest] = {"path": rel, "complete": True}
                if old:
                    remove_file(col, old)
                updated += bool(old)
                added += not old
        except Exception as e:
            failed += 1
            print(f"  failed: {e}")
            logger.exception("Failed to add %s", rel)
        finally:
            cost.finish_query()
            if qc.calls:
                print(f"  cost: ${qc.total_usd:.6f} ({qc.n_calls} requests, "
                      f"{qc.prompt_tokens:,} tokens)")

    files = stored_files(col)
    kept = sorted(f["path"] for f in files.values() if f["path"] not in on_disk)
    for rel in kept:
        print(f"Kept in the store (no longer in documents): {rel}")
    for f in files.values():
        if not f["complete"]:
            print(f"Not finished ({f['stored']} of {f['chunks']} chunks): {f['path']}. "
                  "Run ingest again to continue.")
    print(f"\nDone. Added {added}, updated {updated}, unchanged {unchanged}, failed {failed}.")
    print(f"The store now holds {len(files)} files and {col.count()} chunks.")
    if not on_disk and not files:
        print(f"No documents found. Put your files in: {DOCS_DIR}")


def main():
    setup_logging()
    try:
        run()
    except RAGError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
