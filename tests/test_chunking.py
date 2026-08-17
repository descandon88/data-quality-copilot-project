"""
Unit tests for ingestion/chunk_knowledge_base.py — section-header splitting,
the long-section paragraph/overlap fallback, and end-to-end chunk_document()
against a synthetic markdown file (no real knowledge_base/ doc is read
here, so these tests don't break if the real docs are edited).
"""
from pathlib import Path

from ingestion.chunk_knowledge_base import (
    chunk_document,
    split_into_sections,
    split_long_text,
)


class TestSplitIntoSections:
    def test_no_headers_returns_single_body_section(self):
        sections = split_into_sections("Just some plain text, no headers at all.")
        assert sections == [("Body", "Just some plain text, no headers at all.")]

    def test_splits_on_h2_headers(self):
        body = "## Summary\nThe summary text.\n\n## Impact\nThe impact text."
        sections = split_into_sections(body)
        assert sections == [("Summary", "The summary text."), ("Impact", "The impact text.")]

    def test_preamble_before_first_header_is_captured_separately(self):
        body = "Some intro text.\n\n## Summary\nThe summary text."
        sections = split_into_sections(body)
        assert sections[0] == ("Preamble", "Some intro text.")
        assert sections[1] == ("Summary", "The summary text.")

    def test_empty_section_is_skipped(self):
        body = "## Summary\nreal content\n\n## Empty\n\n## Impact\nmore content"
        sections = split_into_sections(body)
        headings = [h for h, _ in sections]
        assert "Empty" not in headings

    def test_does_not_split_on_h3_headers(self):
        body = "## Summary\nSome text with a ### subheading inside it that shouldn't split."
        sections = split_into_sections(body)
        assert len(sections) == 1
        assert sections[0][0] == "Summary"


class TestSplitLongText:
    def test_short_text_returned_unchanged_as_single_chunk(self):
        text = "short section, well under the word cap"
        assert split_long_text(text, max_words=200) == [text]

    def test_splits_when_over_max_words(self):
        # 3 paragraphs of 100 words each, cap at 150 -> should split into
        # more than one chunk.
        para = " ".join(["word"] * 100)
        text = "\n\n".join([para, para, para])
        chunks = split_long_text(text, max_words=150, overlap=10)
        assert len(chunks) > 1

    def test_overlap_carries_tail_words_into_next_chunk(self):
        para_a = " ".join(f"a{i}" for i in range(100))
        para_b = " ".join(f"b{i}" for i in range(100))
        text = f"{para_a}\n\n{para_b}"
        chunks = split_long_text(text, max_words=150, overlap=10)
        assert len(chunks) == 2
        # The last 10 words of chunk 1 should reappear at the start of chunk 2.
        tail_of_first = " ".join(chunks[0].split()[-10:])
        assert chunks[1].startswith(tail_of_first)

    def test_single_oversized_paragraph_is_not_split_mid_paragraph(self):
        # Current implementation only splits on paragraph boundaries
        # ("\n\n") — a single paragraph longer than max_words with no
        # internal blank line is kept intact rather than cut mid-sentence.
        # This test pins down that documented tradeoff so a future change
        # to the splitting strategy has to be a deliberate edit here, not
        # an accidental regression.
        huge_paragraph = " ".join(["word"] * 500)
        chunks = split_long_text(huge_paragraph, max_words=200)
        assert chunks == [huge_paragraph]


class TestChunkDocument:
    """chunk_document() computes source_file as
    path.relative_to(KB_DIR.parent) — i.e. it assumes the doc lives under
    the real repo's knowledge_base/ directory. To test it against synthetic
    docs (not the real knowledge_base/ files, so these tests don't break
    when someone edits a real postmortem), every test here builds a
    tmp_path/knowledge_base/<folder>/ layout and monkeypatches KB_DIR to
    match it, rather than writing bare files straight into tmp_path."""

    def _write_doc(self, tmp_path: Path, monkeypatch, content: str, folder: str = "postmortems") -> Path:
        kb_dir = tmp_path / "knowledge_base"
        monkeypatch.setattr("ingestion.chunk_knowledge_base.KB_DIR", kb_dir)
        doc_dir = kb_dir / folder
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / "PM-999.md"
        path.write_text(content)
        return path

    def test_basic_two_section_doc(self, tmp_path, monkeypatch):
        content = (
            "---\n"
            "id: PM-999\n"
            "title: Test Incident\n"
            "---\n\n"
            "## Summary\n"
            "A short summary.\n\n"
            "## Impact\n"
            "A short impact statement.\n"
        )
        path = self._write_doc(tmp_path, monkeypatch, content)
        chunks = chunk_document(path, "postmortem")

        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "PM-999__summary"
        assert chunks[0]["doc_id"] == "PM-999"
        assert chunks[0]["doc_type"] == "postmortem"
        assert chunks[0]["title"] == "Test Incident"
        assert chunks[0]["section"] == "Summary"
        assert chunks[0]["text"] == "A short summary."
        assert chunks[0]["embedding_text"] == "[PM-999] Test Incident — Summary\n\nA short summary."
        assert chunks[1]["chunk_id"] == "PM-999__impact"

    def test_falls_back_to_filename_stem_when_frontmatter_id_missing(self, tmp_path, monkeypatch):
        content = "---\ntitle: No Id Doc\n---\n\n## Summary\nsome text\n"
        path = self._write_doc(tmp_path, monkeypatch, content)
        chunks = chunk_document(path, "postmortem")
        assert chunks[0]["doc_id"] == "PM-999"  # falls back to the filename stem

    def test_long_section_gets_multi_piece_chunk_ids(self, tmp_path, monkeypatch):
        long_body = "\n\n".join([" ".join(["word"] * 100)] * 3)  # 300 words, well over MAX_SECTION_WORDS=200
        content = f"---\nid: PM-999\ntitle: Long Doc\n---\n\n## Summary\n{long_body}\n"
        path = self._write_doc(tmp_path, monkeypatch, content)
        chunks = chunk_document(path, "postmortem")

        assert len(chunks) > 1
        assert all(c["chunk_id"].startswith("PM-999__summary-") for c in chunks)
        # piece indices should be distinct and start at 0
        suffixes = sorted(int(c["chunk_id"].rsplit("-", 1)[1]) for c in chunks)
        assert suffixes == list(range(len(chunks)))

    def test_source_file_is_relative_to_repo_root(self, tmp_path, monkeypatch):
        content = "---\nid: PM-999\ntitle: Test\n---\n\n## Summary\ntext\n"
        path = self._write_doc(tmp_path, monkeypatch, content)
        chunks = chunk_document(path, "postmortem")
        # Should read as "knowledge_base/postmortems/PM-999.md", not an
        # absolute tmp_path — that's what makes the stored metadata
        # portable across machines/containers.
        assert chunks[0]["source_file"] == str(Path("knowledge_base") / "postmortems" / "PM-999.md")
