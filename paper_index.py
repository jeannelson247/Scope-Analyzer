"""
paper_index.py - Lightweight local-paper retrieval for Scope Studio.

This is retrieval, not fine-tuning: we index PDFs and text-like files into
small chunks, then pull the most relevant excerpts into the local model's
prompt when the user asks a question.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import os
import re


SUPPORTED_EXTS = {
    ".md", ".markdown", ".m", ".pdf", ".rst", ".tex", ".txt"
}

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "venv", ".venv"}

STOPWORDS = {
    "about", "after", "again", "also", "and", "any", "are", "been", "between",
    "can", "could", "data", "each", "for", "from", "have", "into", "its",
    "just", "maybe", "more", "most", "not", "only", "our", "over", "same",
    "should", "than", "that", "the", "their", "there", "these", "this",
    "those", "through", "using", "was", "were", "what", "when", "where",
    "which", "with", "would", "your",
}


@dataclass
class Chunk:
    source: str
    title: str
    text: str
    tokens: Counter[str]


@dataclass
class PaperIndex:
    folder: str
    chunks: list[Chunk] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    source: str
    title: str
    excerpt: str
    score: float


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9_+\-/.]{2,}", text.lower())
    return [t for t in toks if t not in STOPWORDS]


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs `pypdf`. Install it with "
            "`pip install pypdf`."
        ) from exc
    parts = []
    reader = PdfReader(path)
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def read_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    with open(path, "r", errors="replace") as handle:
        return handle.read()


def iter_documents(folder: str) -> list[str]:
    paths: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def chunk_text(text: str, chunk_size: int = 1200,
               overlap: int = 180) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []
    paras = cleaned.split("\n\n")
    chunks: list[str] = []
    buf = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            carry = buf[-overlap:] if overlap else ""
            buf = f"{carry}\n\n{para}".strip()
        else:
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[max(chunk_size - overlap, 1):]
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def build_index(folder: str) -> PaperIndex:
    idx = PaperIndex(folder=folder)
    docs = iter_documents(folder)
    if not docs:
        raise RuntimeError("No supported papers found in that folder.")
    for path in docs:
        try:
            text = read_document(path)
        except Exception as exc:
            idx.skipped.append(f"{os.path.basename(path)} ({exc})")
            continue
        pieces = chunk_text(text)
        if not pieces:
            idx.skipped.append(f"{os.path.basename(path)} (no extractable text)")
            continue
        idx.files.append(path)
        title = os.path.basename(path)
        for piece in pieces:
            idx.chunks.append(
                Chunk(
                    source=path,
                    title=title,
                    text=piece,
                    tokens=Counter(tokenize(piece)),
                )
            )
    if not idx.chunks:
        details = "; ".join(idx.skipped[:3]) or "No text could be extracted."
        raise RuntimeError(details)
    return idx


def search(index: PaperIndex, query: str, top_k: int = 4) -> list[SearchResult]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    results: list[SearchResult] = []
    for chunk in index.chunks:
        score = 0.0
        text_l = chunk.text.lower()
        for token in q_tokens:
            tf = chunk.tokens.get(token, 0)
            if tf:
                score += 1.0 + math.log1p(tf)
            if token in text_l:
                score += 0.1
        score /= math.sqrt(max(sum(chunk.tokens.values()), 1))
        if score <= 0:
            continue
        excerpt = chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 420:
            excerpt = excerpt[:417] + "..."
        results.append(
            SearchResult(
                source=chunk.source,
                title=chunk.title,
                excerpt=excerpt,
                score=score,
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    deduped: list[SearchResult] = []
    seen: set[tuple[str, str]] = set()
    for item in results:
        key = (item.source, item.excerpt)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped
