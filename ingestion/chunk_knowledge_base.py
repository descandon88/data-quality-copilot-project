"""
Chunks the knowledge_base/ markdown docs (postmortems, rules, contracts) into
embed-ready pieces, prepping for Phase 4 (embedding + pgvector indexing).

Chunking strategy: split by ## section headers, not fixed-size windows.
These documents are already structured in coherent sections (Summary, Root
cause, Resolution, ...) — splitting on that structure keeps each chunk
semantically self-contained, which matters more here than hitting an exact
token count. A section that runs long gets a secondary paragraph-based split
with overlap, so no single chunk balloons past a reasonable size, but the
common case is one chunk per section.

Each chunk is prefixed with its document id/title/section on embedding input
(not stored separately) — "contextualizing" chunks like this measurably
improves retrieval, since a chunk that just says "796 of ~115,139
transactions..." is ambiguous on its own, but "[PM-001] Duplicate loyalty
point credits ... — Impact: 796 of ~115,139 transactions..." is not.

Output: data/processed/kb_chunks.jsonl — one JSON object per chunk. Phase 4
reads this file, embeds `embedding_text`, and writes vectors + metadata into
pgvector. This script does no embedding itself — cleaning/chunking and
embedding are kept as separate steps on purpose, so either can be rerun
independently (e.g. re-chunk after editing a doc without re-embedding
everything, or swap embedding models without re-chunking).

Run inside the app container:
    docker compose exec app python ingestion/chunk_knowledge_base.py
"""
import json
import re
from pathlib import Path

import frontmatter

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb_chunks.jsonl"

DOC_TYPE_BY_FOLDER = {
    "postmortems": "postmortem",
    "rules": "rule",
    "contracts": "contract",
}

MAX_SECTION_WORDS = 200
OVERLAP_WORDS = 30

SECTION_HEADER_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def split_into_sections(body: str):
    """Splits markdown body into (heading, text) pairs on '## ' headers.
    Content before the first '## ' header (if any) is returned under the
    heading 'Preamble'."""
    matches = list(SECTION_HEADER_RE.finditer(body))
    if not matches:
        return [("Body", body.strip())]

    sections = []
    if matches[0].start() > 0:
        preamble = body[: matches[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))

    return sections


def split_long_text(text: str, max_words: int = MAX_SECTION_WORDS, overlap: int = OVERLAP_WORDS):
    """Paragraph-based split with word-count overlap, only invoked when a
    section exceeds max_words. Keeps paragraphs intact where possible rather
    than cutting mid-sentence."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        para_len = len(para.split())
        if current and current_len + para_len > max_words:
            chunks.append("\n\n".join(current))
            # carry the tail of the previous chunk forward as overlap
            overlap_words = " ".join(current[-1].split()[-overlap:])
            current = [overlap_words] if overlap_words else []
            current_len = len(overlap_words.split())
        current.append(para)
        current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text]


def chunk_document(path: Path, doc_type: str):
    post = frontmatter.load(path)
    meta = dict(post.metadata)
    doc_id = meta.get("id", path.stem)
    title = meta.get("title", path.stem)

    sections = split_into_sections(post.content)

    chunks = []
    for section_idx, (heading, text) in enumerate(sections):
        pieces = split_long_text(text)
        for piece_idx, piece in enumerate(pieces):
            chunk_id = f"{doc_id}__{heading.lower().replace(' ', '-')}"
            if len(pieces) > 1:
                chunk_id += f"-{piece_idx}"

            embedding_text = f"[{doc_id}] {title} — {heading}\n\n{piece}"

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "title": title,
                "section": heading,
                "text": piece,
                "embedding_text": embedding_text,
                "metadata": meta,
                "source_file": str(path.relative_to(KB_DIR.parent)),
            })
    return chunks


def main():
    all_chunks = []
    for folder, doc_type in DOC_TYPE_BY_FOLDER.items():
        folder_path = KB_DIR / folder
        if not folder_path.exists():
            continue
        for md_file in sorted(folder_path.glob("*.md")):
            all_chunks.extend(chunk_document(md_file, doc_type))

    if not all_chunks:
        print(f"No markdown docs found under {KB_DIR}. Nothing to chunk yet.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, default=str) + "\n")

    by_type = {}
    for c in all_chunks:
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1

    print(f"Wrote {len(all_chunks)} chunks -> {OUTPUT_PATH}")
    for doc_type, count in by_type.items():
        print(f"  {doc_type}: {count} chunks")


if __name__ == "__main__":
    main()