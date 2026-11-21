# Look inside the vector store (ChromaDB, chroma_db_global/).
#
#   python -m ext_rag.show_db
#   python -m ext_rag.show_db --file "name.pdf"
#   python -m ext_rag.show_db --chunk ID
#   python -m ext_rag.show_db --collection global_hybrid_collection

import sys
import logging
import argparse

import numpy as np

from hybrid_rag.logging_setup import setup_logging
from ext_rag.rag_common import RAGError, STORE_DIR, COLLECTION, _client, fail

logger = logging.getLogger("rag.show_db")


def folder_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def group_files(metadatas):
    files = {}
    for m in metadatas:
        m = m or {}
        key = m.get("hash") or m.get("source") or "(no file name)"
        f = files.setdefault(
            key,
            {
                "path": m.get("path") or m.get("source") or key,
                "file_type": m.get("file_type", ""),
                "added": m.get("added", ""),
                "chunks": m.get("chunks"),
                "skipped": m.get("skipped", 0),
                "stored": 0,
            },
        )
        f["stored"] += 1
    return files


def file_state(f):
    if f["chunks"] is None:
        return f"{f['stored']} chunks"
    if f["stored"] + f["skipped"] < f["chunks"]:
        return f"NOT FINISHED ({f['stored']} saved)"
    return f"complete, {f['skipped']} skipped" if f["skipped"] else "complete"


def overview(client):
    collections = client.list_collections()
    print(f"Store: {STORE_DIR}  ({folder_size(STORE_DIR) / 1e6:.1f} MB on disk)")
    print(f"Current Ext backend collection: {COLLECTION}\n")
    if not collections:
        print("No collections yet. Upload PDFs in the app or run: python -m ext_rag.ingest")
    for c in collections:
        col = client.get_collection(c.name)
        settings = ", ".join(f"{k}={v}" for k, v in (col.metadata or {}).items())
        files = group_files(col.get(include=["metadatas"])["metadatas"])
        print(f"=== {col.name}  ({col.count()} chunks, {len(files)} files)")
        print(f"    settings: {settings}")
        for h, f in sorted(files.items(), key=lambda x: x[1]["path"]):
            if f["chunks"] is None:
                print(f"    {f['path']:50} {file_state(f)}")
                continue
            print(f"    {f['path']:50} {f['file_type']:5} {f['chunks']:4} chunks  {file_state(f):22} "
                  f"added {f['added']}  hash {h[:12]}")
        print()


def show_file(col, name):
    got = col.get(where={"path": name}, include=["metadatas", "documents"])
    if not got["ids"]:
        got = col.get(where={"source": name}, include=["metadatas", "documents"])
    if not got["ids"]:
        fail(f'"{name}" is not in {col.name}. See the files with: python -m ext_rag.show_db')
    rows = list(zip(got["ids"], got["metadatas"], got["documents"]))
    if rows[0][1] and "chunk" in rows[0][1]:
        rows.sort(key=lambda r: r[1]["chunk"])
    print(f"{name} in {col.name}: {len(rows)} chunks saved\n")
    print(f"{'IDX':>5}  {'LOCATION':14} {'WORDS':>5}  {'ID':22} TEXT")
    for idx, (cid, m, text) in enumerate(rows):
        m = m or {}
        loc = m.get("location") or (f"page {m['page']}" if "page" in m else "-")
        short_id = f"{cid[:8]}..{cid[-6:]}" if len(cid) > 16 else cid
        preview = " ".join(text.split())[:70]
        print(f"{m.get('chunk', idx):5}  {loc:14} {len(text.split()):5}  {short_id:22} {preview}...")
    print(f"\nFull chunk: python -m ext_rag.show_db --collection {col.name} --chunk {rows[0][0]}")


def show_chunk(col, chunk_id):
    got = col.get(ids=[chunk_id], include=["metadatas", "documents", "embeddings"])
    if not got["ids"]:
        fail(f'No chunk with id "{chunk_id}" in {col.name}.')
    m, text, vec = got["metadatas"][0], got["documents"][0], np.asarray(got["embeddings"][0])
    print(f"id:         {chunk_id}\ncollection: {col.name}")
    for k, v in (m or {}).items():
        print(f"{k + ':':11} {v}")
    print(f"vector:     {len(vec)} numbers, length {np.linalg.norm(vec):.3f}, starts "
          f"[{', '.join(f'{x:.4f}' for x in vec[:8])}, ...]")
    print(f"\n--- text ({len(text.split())} words) ---\n{text}")


def run():
    parser = argparse.ArgumentParser(description="Look inside the ChromaDB vector store")
    parser.add_argument("--file", metavar="NAME", help="list every chunk of this file")
    parser.add_argument("--chunk", metavar="ID", help="show one chunk in full")
    parser.add_argument("--collection", default=COLLECTION,
                        help=f"collection for --file / --chunk (default: {COLLECTION})")
    args = parser.parse_args()

    if not STORE_DIR.exists():
        fail("The vector store doesn't exist yet. Upload PDFs in the app or run: python -m ext_rag.ingest")
    client = _client()
    if not (args.file or args.chunk):
        overview(client)
        return
    if args.collection not in [c.name for c in client.list_collections()]:
        fail(f'There is no collection "{args.collection}". See them with: python -m ext_rag.show_db')
    col = client.get_collection(args.collection)
    if args.file:
        show_file(col, args.file)
    if args.chunk:
        show_chunk(col, args.chunk)


def main():
    setup_logging()
    try:
        run()
    except RAGError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
