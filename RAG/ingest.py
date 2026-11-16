# Add every supported file in RAG/documents/ to the vector store (ChromaDB, chroma_db_global/).
#
# Usage:
#   python -m RAG.ingest                 add new files and re-embed changed ones
#   python -m RAG.ingest --remove FILE   delete one document from the store
#   python -m RAG.ingest --rebuild       delete the store and embed everything again
#
# Each file is embedded only once. Files deleted from documents/ stay in the store,
# so questions about them can still be answered.
#
# Supported: .pdf .docx .pptx .txt .md .csv .html .htm
# Files can be on any topic and can sit in subfolders; folder names don't matter.

import sys, csv, time, hashlib, logging, argparse
from datetime import datetime

from RAG.rag_common import (RAGError, DOCS_DIR, STORE_DIR, COLLECTION, CHUNK_WORDS, OVERLAP_WORDS, BACKEND, BATCH,
                        EMBED_MODEL, embed_texts, open_store, store_exists, delete_store,
                        stored_files, fail)
from RAGs.logger_config import setup_logging

logger = logging.getLogger("rag.ingest")


# ---------- Loaders: each returns (kind, [(label, text), ...]) ----------
# "kind" names the location unit (page, slide, line, ...); "label" is that
# unit's number, or a section heading for Word files.

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
        # Label each paragraph with the heading it falls under
        kind, section = "section", "Beginning"
        for p in paras:
            if is_heading(p):
                section = p.text.strip()[:60]
            units.append((section, p.text))
    else:
        # No headings: number the paragraphs instead
        kind = "paragraph"
        units = [(i, p.text) for i, p in enumerate(paras, 1)]

    # Tables (text only, one row per line)
    for t, table in enumerate(doc.tables, 1):
        label = f"Table {t}" if kind == "section" else len(units) + 1
        for row in table.rows:
            cells = []
            for c in row.cells:                      # merged cells repeat, skip duplicates
                txt = c.text.strip()
                if txt and (not cells or cells[-1] != txt):
                    cells.append(txt)
            if cells:
                units.append((label, " | ".join(cells)))
    return kind, units


def load_pptx(path):
    from pptx import Presentation
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
        # "column: value" keeps the meaning of each cell for the search
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        units.append((i, "; ".join(pairs) + "."))
    return "row", units


def load_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(read_text(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    return "line", list(enumerate(lines, 1))


LOADERS = {
    ".pdf": load_pdf, ".docx": load_docx, ".pptx": load_pptx,
    ".txt": load_txt, ".md": load_txt, ".csv": load_csv,
    ".html": load_html, ".htm": load_html,
}


# ---------- Chunking ----------
def format_location(kind, labels):
    labels = list(dict.fromkeys(labels))            # unique, in order
    if all(isinstance(x, int) for x in labels):
        lo, hi = min(labels), max(labels)
        return f"{kind} {lo}" if lo == hi else f"{kind}s {lo}-{hi}"
    shown = "; ".join(str(x) for x in labels[:3])
    return f"{kind}: {shown}" + (" ..." if len(labels) > 3 else "")


def make_chunks(kind, units):
    """Sliding window of CHUNK_WORDS words with OVERLAP_WORDS overlap.
    Each word remembers its label, so each chunk knows where it came from."""
    words, labels = [], []
    for label, text in units:
        w = text.split()
        words += w
        labels += [label] * len(w)

    chunks, start, step = [], 0, CHUNK_WORDS - OVERLAP_WORDS
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        chunks.append((format_location(kind, labels[start:end]),
                       " ".join(words[start:end])))
        if end == len(words):
            break
        start += step
    logger.debug("%d words in %d %ss -> %d chunks of up to %d words (%d overlap)",
                 len(words), len(units), kind, len(chunks), CHUNK_WORDS, OVERLAP_WORDS)
    return chunks, len(words)


# ---------- Store maintenance ----------
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_file(col, digest):
    col.delete(where={"hash": digest})
    logger.debug("Deleted the chunks of %s from the store", digest[:12])


def stored_chunks(col, digest):
    """Numbers of the chunks of this file that are already in the store."""
    return {m["chunk"] for m in col.get(where={"hash": digest}, include=["metadatas"])["metadatas"]}


def add_file(col, path, rel, digest, done=frozenset()):
    """Embed and save the file's chunks, except those in done (already saved by an earlier
    run that was cut short). Chunks are saved BATCH at a time as soon as they are embedded,
    so if the run stops (daily limit, crash, Ctrl+C) the next run only does the rest."""
    loader = LOADERS[path.suffix.lower()]
    started = time.perf_counter()
    kind, units = loader(path)
    logger.debug("%s read %s: %d %ss in %.1fs", loader.__name__, rel, len(units), kind,
                 time.perf_counter() - started)
    chunks, n_words = make_chunks(kind, units)
    if n_words < 20:
        print(f"  skipped: almost no text found "
              f"({'scanned PDF? needs OCR' if path.suffix.lower() == '.pdf' else 'empty file?'})")
        logger.warning("Skipped %s: only %d words found", rel, n_words)
        return 0

    todo = [i for i in range(len(chunks)) if i not in done]
    print(f"  {n_words} words -> {len(chunks)} chunks" +
          (f", {len(done)} already saved" if done else "") + f", embedding {len(todo)}...")

    # The ids come from the file's content, so writing the same file again overwrites its
    # chunks instead of adding copies. Until every chunk is saved, the file counts as
    # incomplete (fewer chunks stored than "chunks" says).
    added = datetime.now().isoformat(timespec="seconds")
    for start in range(0, len(todo), BATCH):
        part = todo[start:start + BATCH]
        vectors = embed_texts([chunks[i][1] for i in part])
        col.upsert(
            ids=[f"{digest}:{i}" for i in part],
            embeddings=vectors,
            documents=[chunks[i][1] for i in part],
            metadatas=[{"path": rel, "location": chunks[i][0], "hash": digest,
                        "file_type": path.suffix.lower().lstrip("."), "added": added,
                        "chunk": i, "chunks": len(chunks)}
                       for i in part])
        saved = len(done) + start + len(part)
        print(f"  saved {saved}/{len(chunks)} chunks")
        logger.debug("Saved %d chunks of %s, %d of %d done", len(part), rel, saved, len(chunks))
    logger.info("Embedded %s (%s): %d words -> %d chunks", rel, digest[:12], n_words, len(chunks))
    return len(chunks)


def run():
    parser = argparse.ArgumentParser(description="Add documents to the vector store")
    parser.add_argument("--rebuild", action="store_true", help="start from an empty store")
    parser.add_argument("--force", action="store_true",
                        help="with --rebuild: also lose documents that are no longer in the folder")
    parser.add_argument("--remove", metavar="FILE",
                        help="delete one document from the store (path as shown by ask.py --list)")
    args = parser.parse_args()

    DOCS_DIR.mkdir(exist_ok=True)
    on_disk = {}
    for p in sorted(DOCS_DIR.rglob("*")):
        if not p.is_file() or p.name.startswith(("~$", ".")):
            continue                                       # skip Office temp files
        if p.suffix.lower() in LOADERS:
            on_disk[p.relative_to(DOCS_DIR).as_posix()] = p
        else:
            print(f"Unsupported file type, ignored: {p.name}")
    logger.info("Found %d supported files in %s", len(on_disk), DOCS_DIR)

    if args.rebuild and store_exists():
        # The store is the only copy of documents that were removed from the folder
        only_in_store = sorted({f["path"] for f in stored_files(open_store(check=False)).values()}
                               - set(on_disk))
        if only_in_store and not args.force:
            fail("These documents are in the store but no longer in RAG/documents, so a rebuild "
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
            fail(f'"{args.remove}" is not in the store. See the list with: python -m RAG.ask --list')
        for h in hashes:
            remove_file(col, h)
        print(f"Removed from the store: {args.remove}")
        logger.info("Removed %s on request", args.remove)
        return

    digests = {rel: file_hash(path) for rel, path in on_disk.items()}

    # Imports that were cut short (daily limit, crash, Ctrl+C) are continued below. Only
    # drop them if that file is no longer in the folder (deleted or changed in between).
    for h, f in list(stored.items()):
        if not f["complete"] and h not in digests.values():
            remove_file(col, h)
            del stored[h]
            logger.warning("Removed the unfinished import of %s (%d of %d chunks): the file "
                           "is no longer in the folder", f["path"], f["stored"], f["chunks"])

    # Every file is embedded once, identified by its content (hash), not its name:
    #  - a file already in the store is skipped, even if it was renamed, moved or copied
    #  - a file whose content changed replaces its old version
    #  - a file deleted from the folder STAYS in the store, so questions about it still
    #    work (use --remove to delete it from the store)
    by_path = {f["path"]: h for h, f in stored.items() if f["complete"]}
    added = updated = unchanged = failed = 0
    for rel, path in on_disk.items():
        digest = digests[rel]
        if digest in stored and stored[digest]["complete"]:
            unchanged += 1
            if stored[digest]["path"] != rel:
                print(f"Already in the store as {stored[digest]['path']}, not embedded again: {rel}")
            logger.debug("Unchanged, not embedded again: %s (%s)", rel, digest[:12])
            continue
        old = by_path.get(rel)
        done = stored_chunks(col, digest) if digest in stored else set()
        print(f"{'Continuing' if done else 'Updating' if old else 'Adding'}: {rel}")
        try:
            if add_file(col, path, rel, digest, done):
                stored[digest] = {"path": rel, "complete": True}   # a later identical copy is skipped
                if old:
                    remove_file(col, old)          # only once the new version is safely stored
                    logger.info("Replaced the old version of %s (%s)", rel, old[:12])
                updated += bool(old)
                added += not old
        except Exception as e:
            # Nothing is written before the file has been read, and chunks already
            # embedded are kept, so the next run continues from there
            failed += 1
            print(f"  failed: {e}")
            logger.exception("Failed to add %s", rel)

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
    logger.info("Done: added %d, updated %d, unchanged %d, failed %d, %d kept that are no longer "
                "in the folder. Store: %d files, %d chunks",
                added, updated, unchanged, failed, len(kept), len(files), col.count())
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