r"""Intelligent Novel Chaptering and Chunk Partitioning Engine.

Handles full 100,000+ word book manuscripts with:
1. Regex detection of Chapter, Act, Scene, Prologue, Epilogue, Book headings.
2. Markdown / LaTeX heading detection (`# Chapter 1`, `\chapter{...}`).
3. Fallback paragraph-boundary windowing for unformatted prose manuscripts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


CHAPTER_HEADING_REGEX = re.compile(
    r'(?im)^(?:\s*#{1,3}\s*)?(?:Chapter|CHAPTER|Section|SECTION|Part|PART|Act|ACT|Book|BOOK|Prologue|PROLOGUE|Epilogue|EPILOGUE|Scene|SCENE)\s*([0-9IVXLCDM]+|[A-Za-z]+)?(?::|\.|\-|\s)*(.*?)$'
)


def split_manuscript_into_chapters(
    text: str,
    target_chunk_words: int = 2500,
    min_chunk_words: int = 10,
) -> List[Dict[str, Any]]:
    """Partition a long manuscript into coherent chapters or structured chunks.

    Args:
        text: Raw full text of the manuscript.
        target_chunk_words: Desired size in words if splitting by windows.
        min_chunk_words: Minimum words to consider a chunk valid.

    Returns:
        List of dicts: [{"chapter_index": 1, "title": "Chapter 1: The Beginning", "text": "...", "word_count": 3200}, ...]
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()
    matches = list(CHAPTER_HEADING_REGEX.finditer(clean_text))

    # If 2 or more chapter headings are detected, split by matched boundaries
    if len(matches) >= 2:
        chapters = []
        for i, m in enumerate(matches):
            start_pos = m.start()
            end_pos = matches[i + 1].start() if (i + 1) < len(matches) else len(clean_text)
            
            raw_title = m.group(0).strip().replace("#", "").strip()
            chap_body = clean_text[start_pos:end_pos].strip()
            word_count = len(chap_body.split())

            if word_count >= min_chunk_words or i == len(matches) - 1:
                chapters.append({
                    "chapter_index": len(chapters) + 1,
                    "title": raw_title[:60],
                    "text": chap_body,
                    "word_count": word_count,
                })
        if chapters:
            return chapters

    # Fallback: Partition by logical paragraph chunks around target_chunk_words
    # PDF text often has single \n instead of \n\n, so normalize first
    normalized = re.sub(r'\r\n', '\n', clean_text)
    # Collapse single newlines that aren't paragraph breaks (not preceded/followed by blank line)
    normalized = re.sub(r'(?<!\n)\n(?!\n)', ' ', normalized)
    # Now split on actual paragraph boundaries (2+ newlines)
    raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', normalized) if p.strip()]

    # Filter out PDF noise: page numbers, headers, very short fragments
    paragraphs = []
    for p in raw_paragraphs:
        # Skip standalone numbers (page numbers), very short noise
        if re.match(r'^\d{1,4}$', p.strip()):
            continue
        if re.match(r'^Page\s+\d+', p.strip(), re.IGNORECASE):
            continue
        word_count = len(p.split())
        if word_count < 3:
            continue
        # Skip lines that are mostly non-alphabetic
        alpha_ratio = sum(1 for c in p if c.isalpha()) / max(1, len(p))
        if alpha_ratio < 0.4:
            continue
        paragraphs.append(p)

    if not paragraphs:
        paragraphs = [clean_text]

    chunks = []
    current_paras: List[str] = []
    current_words = 0
    chunk_index = 1

    for p in paragraphs:
        p_word_count = len(p.split())
        current_paras.append(p)
        current_words += p_word_count

        if current_words >= target_chunk_words:
            chunks.append({
                "chapter_index": chunk_index,
                "title": f"Section {chunk_index} (Words {sum(c['word_count'] for c in chunks) + 1:,} - {sum(c['word_count'] for c in chunks) + current_words:,})",
                "text": "\n\n".join(current_paras),
                "word_count": current_words,
            })
            chunk_index += 1
            current_paras = []
            current_words = 0

    if current_paras:
        # Merge tiny trailing chunks into the previous one if too small
        if current_words < min_chunk_words and chunks:
            chunks[-1]["text"] += "\n\n" + "\n\n".join(current_paras)
            chunks[-1]["word_count"] += current_words
        else:
            chunks.append({
                "chapter_index": chunk_index,
                "title": f"Section {chunk_index} (Final Section)",
                "text": "\n\n".join(current_paras),
                "word_count": current_words,
            })

    return chunks if chunks else [{
        "chapter_index": 1,
        "title": "Full Manuscript",
        "text": clean_text,
        "word_count": len(clean_text.split()),
    }]


def split_manuscript_into_pages(
    text: str,
    target_words_per_page: int = 280,
) -> List[Dict[str, Any]]:
    """Partition text into editorial standard manuscript pages (~250-350 words / standard A4 page).
    
    If text contains explicit PDF page delimiters ('\\x0c' or '<!-- PAGE_BREAK -->'),
    it splits on those; otherwise, it partitions by paragraphs into standard ~280-word pages.
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()

    # 1. Check for physical PDF / EPUB page breaks
    if "\x0c" in clean_text or "<!-- PAGE_BREAK -->" in clean_text:
        raw_pages = re.split(r'\x0c|<!-- PAGE_BREAK -->', clean_text)
        pages = []
        for idx, page_str in enumerate(raw_pages):
            p_clean = page_str.strip()
            if not p_clean:
                continue
            words = len(p_clean.split())
            if words >= 5:
                pages.append({
                    "page_number": len(pages) + 1,
                    "title": f"Page {len(pages) + 1}",
                    "text": p_clean,
                    "word_count": words,
                    "char_count": len(p_clean),
                })
        if pages:
            return pages

    # 2. Split into clean paragraphs
    normalized = clean_text.replace('\r\n', '\n').strip()
    
    if '\n\n' in normalized:
        raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', normalized) if p.strip()]
    elif '\n' in normalized:
        raw_paragraphs = [l.strip() for l in normalized.split('\n') if l.strip()]
    else:
        raw_paragraphs = [normalized]

    paragraphs = []
    for p in raw_paragraphs:
        if re.match(r'^\d{1,4}$', p.strip()):
            continue
        if re.match(r'^Page\s+\d+', p.strip(), re.IGNORECASE):
            continue
        if len(p.split()) < 3:
            continue
            
        # If a single paragraph is very long (> 150 words or > 6 sentences), split into smaller cohesive paragraphs
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', p) if s.strip()]
        if len(sents) > 6:
            for i in range(0, len(sents), 5):
                sub_s = sents[i:i + 5]
                if sub_s:
                    paragraphs.append(" ".join(sub_s))
        else:
            paragraphs.append(p)

    if not paragraphs:
        paragraphs = [clean_text]

    pages = []
    current_paras: List[str] = []
    current_words = 0
    page_num = 1

    for p in paragraphs:
        p_words = len(p.split())
        current_paras.append(p)
        current_words += p_words

        if current_words >= target_words_per_page:
            pages.append({
                "page_number": page_num,
                "title": f"Page {page_num}",
                "text": "\n\n".join(current_paras),
                "word_count": current_words,
                "char_count": sum(len(x) for x in current_paras),
            })
            page_num += 1
            current_paras = []
            current_words = 0

    if current_paras:
        if current_words < 50 and pages:
            # Merge tiny trailing fragment into previous page
            pages[-1]["text"] += "\n\n" + "\n\n".join(current_paras)
            pages[-1]["word_count"] += current_words
            pages[-1]["char_count"] += sum(len(x) for x in current_paras)
        else:
            pages.append({
                "page_number": page_num,
                "title": f"Page {page_num}",
                "text": "\n\n".join(current_paras),
                "word_count": current_words,
                "char_count": sum(len(x) for x in current_paras),
            })

    return pages if pages else [{
        "page_number": 1,
        "title": "Page 1",
        "text": clean_text,
        "word_count": len(clean_text.split()),
        "char_count": len(clean_text),
    }]
