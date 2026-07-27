"""
AinSeba - Metadata-Aware Document Chunker
Splits Bangladesh law documents into semantically meaningful chunks
while preserving structural metadata (Act -> Part -> Chapter -> Section).

FIXED VERSION
-------------
Four bugs corrected versus the original:

1. Section detection only matched the literal form "Section 42. Title".
   Real bdlaws.minlaw.gov.bd PDFs use "42. Title". NUMBERED_SECTION_PATTERN
   existed but was never called, so Labour Act / Muslim Family / Environment
   Act produced zero section markers.

2. PART/CHAPTER patterns anchored with `^` but extracted lines carry a leading
   space (" CHAPTER XV"), so most chapter headings were missed
   (7 of 27 in the Labour Act, 2 of 51 in the Penal Code).

3. No table-of-contents stripping. The first few hundred lines of these PDFs
   are a TOC whose entries look exactly like section headings, which poisons
   retrieval with contents-page chunks.

4. _split_large_section could emit a chunk larger than the token budget when a
   single paragraph exceeded it, producing 18k-22k token chunks that exceed the
   8191-token limit of text-embedding-3-small and get dropped at index time.
   Splitting is now paragraph -> sentence -> hard token slice.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================
# Token Counter
# ============================================

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (OpenAI tokenizer)."""
        return len(_encoding.encode(text))

    def slice_by_tokens(text: str, max_tokens: int) -> list[str]:
        """Hard-split text into pieces of at most max_tokens tokens."""
        ids = _encoding.encode(text)
        return [
            _encoding.decode(ids[i : i + max_tokens])
            for i in range(0, len(ids), max_tokens)
        ]

except Exception:
    # tiktoken not installed OR can't download BPE data (network restricted)
    # Fall back to word-based approximation (~1.3 tokens per word for English)
    logger.warning(
        "tiktoken unavailable. Using approximate token counting. "
        "Install tiktoken and ensure network access for exact counts."
    )

    def count_tokens(text: str) -> int:
        """Approximate token count based on word splitting (~1.3 tokens/word)."""
        if not text:
            return 0
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def slice_by_tokens(text: str, max_tokens: int) -> list[str]:
        """Approximate hard split on word boundaries."""
        words = text.split()
        per = max(1, int(max_tokens / 1.3))
        return [" ".join(words[i : i + per]) for i in range(0, len(words), per)]


# Absolute ceiling for any single chunk. text-embedding-3-small accepts 8191
# tokens; staying well below leaves room for tokenizer drift on Bangla text.
MAX_EMBED_TOKENS = 4000


# ============================================
# Data Models
# ============================================

@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk for retrieval."""
    act_name: str
    act_id: str
    part: str = ""
    chapter: str = ""
    section_number: str = ""
    section_title: str = ""
    category: str = ""
    year: int = 0
    language: str = "english"
    page_numbers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "act_name": self.act_name,
            "act_id": self.act_id,
            "part": self.part,
            "chapter": self.chapter,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "category": self.category,
            "year": self.year,
            "language": self.language,
            "page_numbers": self.page_numbers,
        }


@dataclass
class Chunk:
    """A single document chunk with text and metadata."""
    chunk_id: str
    text: str
    token_count: int
    metadata: ChunkMetadata
    chunk_index: int = 0  # Position in the document

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            **self.metadata.to_dict(),
        }


# ============================================
# Section Candidate Detection
# ============================================

# A section heading, tolerating a marginal-note prefix.
#
# Old bdlaws PDFs (Penal Code 1860, Tenancy Act 1950) print the section title
# in a left margin column. PyMuPDF splices that column into the body line, so
# the heading reads "Punishment for 379. Whoever commits theft..." rather than
# "379. Whoever commits theft...". In the Penal Code body, 86 headings are
# displaced this way against only 83 at line start -- an anchored pattern
# silently loses half the statute, including s.379, the punishment for theft.
_SECTION_CANDIDATE = re.compile(
    r"(?m)^[ \t]*(?:(?P<margin>[A-Za-z][A-Za-z0-9 ,.\-\']{0,45}?)[ \t]+)?"
    r"(?P<num>\d+[A-Za-z]?)\.[ \t]+(?=[A-Z\"(\u201c])"
)

_TOC_ENTRY = re.compile(r"^\s*(\d+[A-Za-z]?)\.\s+(\S.*?)\s*$")


def _sort_key(num: str) -> tuple[int, str]:
    """Sort '379A' after '379' and before '380'."""
    m = re.match(r"(\d+)([A-Za-z]?)", num)
    return (int(m.group(1)), m.group(2).upper()) if m else (0, "")


def longest_ascending_run(candidates: list[dict]) -> list[dict]:
    """
    Keep the largest subset of candidates whose numbers ascend.

    A simple "reject anything not greater than the last kept" walk is too
    fragile: one stray high number early in the document rejects every genuine
    section after it. In the Environment Act a spurious "12A" appearing second
    discarded sections 3 through 12. Solving for the longest strictly ascending
    subsequence instead discards the outlier and keeps the real run.
    """
    if not candidates:
        return []

    import bisect

    keys = [c["sort"] for c in candidates]
    tails: list[int] = []
    tail_keys: list[tuple] = []
    prev = [-1] * len(keys)

    for i, key in enumerate(keys):
        j = bisect.bisect_left(tail_keys, key)
        prev[i] = tails[j - 1] if j > 0 else -1
        if j == len(tails):
            tails.append(i)
            tail_keys.append(key)
        else:
            tails[j] = i
            tail_keys[j] = key

    out = []
    i = tails[-1]
    while i != -1:
        out.append(candidates[i])
        i = prev[i]
    return out[::-1]


def find_section_candidates(text: str) -> list[dict]:
    """
    Locate every plausible section heading, mid-line ones included.

    Returns dicts with: pos, num, sort, margin
    """
    out = []
    for m in _SECTION_CANDIDATE.finditer(text):
        num = m.group("num")
        out.append({
            "pos": m.start(),
            "num": num,
            "sort": _sort_key(num),
            "margin": (m.group("margin") or "").strip(" ,.-"),
        })
    return out


# ============================================
# Table-of-Contents Stripper
# ============================================

def extract_toc_titles(text: str) -> dict[str, str]:
    """
    Harvest section titles from the contents pages.

    The TOC lists them cleanly ("379. Punishment for theft."), whereas the body
    copy is fragmented by the marginal-note splice. Using the TOC gives clean
    citation metadata for free.
    """
    titles: dict[str, str] = {}
    for line in text.split("\n"):
        m = _TOC_ENTRY.match(line)
        if not m:
            continue
        num, title = m.group(1), m.group(2).rstrip(".").strip()
        # Only keep entries that read like titles, not statutory prose.
        if 2 < len(title) <= 120 and num not in titles:
            titles[num] = title
    return titles


def strip_table_of_contents(text: str) -> tuple[str, int]:
    """
    Remove the leading contents pages from a law document.

    A TOC is a run of section numbers that climbs high and then resets near 1
    when the body starts. Detection uses the marginal-note-tolerant matcher, so
    it now works on documents whose body headings are not at line start -- the
    Penal Code previously defeated this check entirely.

    Returns:
        (body_text, lines_removed)
    """
    lines = text.split("\n")
    line_starts = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1

    candidates = find_section_candidates(text)
    if len(candidates) < 10:
        return text, 0

    import bisect
    peak = 0
    resets: list[int] = []
    for c in candidates:
        n = c["sort"][0]
        if n > peak:
            peak = n
        elif n <= 2 and peak >= 10:
            resets.append(c["pos"])
            peak = n

    if not resets:
        return text, 0

    total = len(lines)
    for pos in resets:
        line_idx = bisect.bisect_right(line_starts, pos) - 1
        if (total - line_idx) / total >= 0.5:
            logger.info(
                f"Stripped table of contents: removed {line_idx} lines "
                f"({line_idx / total:.0%} of document)"
            )
            return "\n".join(lines[line_idx:]), line_idx

    return text, 0


# ============================================
# Structure Parser
# ============================================

class LegalStructureParser:
    """
    Parses the hierarchical structure of Bangladesh law documents.
    Detects: PART -> CHAPTER -> Section boundaries.

    All patterns tolerate leading whitespace, which PDF extraction routinely
    introduces, and section headings are matched even when a marginal note has
    been spliced in front of them.
    """

    PART_PATTERN = re.compile(
        r"(?m)^[ \t]*(?:PART)\s+([IVXLCDM]+|\d+)\s*[:\-\u2014.]?\s*(.*?)$",
        re.IGNORECASE,
    )
    CHAPTER_PATTERN = re.compile(
        r"(?m)^[ \t]*(?:CHAPTER)\s+([IVXLCDM]+|\d+)\s*[:\-\u2014.]?\s*(.*?)$",
        re.IGNORECASE,
    )

    _NOISE = re.compile(r"^(page|contents?|section content)\b", re.IGNORECASE)

    @staticmethod
    def _heading_title(inline: str, text: str, match_end: int) -> str:
        """
        Resolve a PART/CHAPTER title.

        In these PDFs the heading number and its title sit on separate lines
        (" CHAPTER IX" then " WORKING HOUR AND LEAVE"), so an inline title is
        usually absent. When one is present it is only trusted if it looks like
        a heading rather than the start of a sentence -- otherwise a wrapped
        definitions line gets absorbed as the chapter name.
        """
        inline = inline.strip(" :-\u2014.")
        if inline and len(inline) <= 60 and not inline.startswith("("):
            letters = [c for c in inline if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
                return inline

        for line in text[match_end : match_end + 300].split("\n")[1:4]:
            line = line.strip()
            if not line:
                continue
            letters = [c for c in line if c.isalpha()]
            if letters and len(line) <= 80 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
                return line
            break

        return ""

    def find_sections(
        self, text: str, toc_titles: Optional[dict] = None
    ) -> list[dict]:
        """
        Find all structural boundaries in the text.

        Section candidates are filtered to an ascending run. Statutes number
        sequentially, so a candidate whose number goes backwards is almost
        always a numbered list item or a cross-reference rather than a heading.
        This is what makes mid-line matching safe.

        Returns:
            List of dicts with keys: type, number, title, start_pos, end_pos
        """
        toc_titles = toc_titles or {}
        markers: list[dict] = []

        for match in self.PART_PATTERN.finditer(text):
            markers.append({
                "type": "part",
                "number": match.group(1).strip(),
                "title": self._heading_title(match.group(2), text, match.end()),
                "start_pos": match.start(),
            })

        for match in self.CHAPTER_PATTERN.finditer(text):
            markers.append({
                "type": "chapter",
                "number": match.group(1).strip(),
                "title": self._heading_title(match.group(2), text, match.end()),
                "start_pos": match.start(),
            })

        heading_positions = {m["start_pos"] for m in markers}

        candidates = [
            c for c in find_section_candidates(text)
            if c["pos"] not in heading_positions
        ]
        ascending = longest_ascending_run(candidates)
        rejected = len(candidates) - len(ascending)
        kept = len(ascending)

        for cand in ascending:
            title = toc_titles.get(cand["num"]) or cand["margin"]
            if self._NOISE.match(title):
                title = ""

            markers.append({
                "type": "section",
                "number": cand["num"],
                "title": title[:200],
                "start_pos": cand["pos"],
            })

        if rejected:
            logger.debug(
                f"Section candidates: kept {kept}, rejected {rejected} out of sequence"
            )

        markers.sort(key=lambda m: m["start_pos"])

        for i, marker in enumerate(markers):
            if i + 1 < len(markers):
                marker["end_pos"] = markers[i + 1]["start_pos"]
            else:
                marker["end_pos"] = len(text)

        return markers


# ============================================
# Main Chunker
# ============================================

class MetadataAwareChunker:
    """
    Chunks Bangladesh law documents while preserving legal structure metadata.

    Strategy:
    1. Strip the table of contents
    2. Parse document structure (Part -> Chapter -> Section)
    3. Keep each Section as one chunk when it fits the token budget
    4. Split oversized sections at paragraph, then sentence, then token level
    5. Apply overlap between chunks for retrieval continuity
    6. Attach hierarchical metadata to each chunk
    """

    def __init__(
        self,
        chunk_size_tokens: int = 600,
        chunk_overlap_tokens: int = 100,
        min_chunk_tokens: int = 50,
        min_section_tokens: int = 15,
        strip_toc: bool = True,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        # Numbered sections get a much lower floor than loose prose. Penal Code
        # punishment sections are deliberately terse -- s.379 is ~40 tokens --
        # and a 50-token floor discarded exactly the sections users ask about.
        self.min_section_tokens = min_section_tokens
        self.strip_toc = strip_toc
        self.parser = LegalStructureParser()

    def chunk_document(
        self,
        text: str,
        act_name: str,
        act_id: str,
        category: str = "",
        year: int = 0,
        language: str = "english",
    ) -> list[Chunk]:
        """
        Split a full law document into metadata-rich chunks.

        Returns:
            List of Chunk objects with metadata.
        """
        logger.info(f"Chunking document: {act_name}")

        # Step 0: Harvest clean section titles from the contents pages, then
        # drop those pages so they never reach the index.
        toc_titles: dict[str, str] = {}
        if self.strip_toc:
            toc_titles = extract_toc_titles(text)
            text, removed = strip_table_of_contents(text)
            if removed:
                logger.info(
                    f"  Removed {removed} table-of-contents lines "
                    f"({len(toc_titles)} section titles harvested)"
                )
            else:
                toc_titles = {}

        # Step 1: Parse document structure
        markers = self.parser.find_sections(text, toc_titles)

        if not markers:
            logger.warning(
                f"No structural markers found in '{act_name}'. "
                f"Falling back to paragraph-based chunking."
            )
            return self._fallback_chunk(
                text, act_name, act_id, category, year, language
            )

        logger.info(f"Found {len(markers)} structural markers")

        # Step 2: Build chunks respecting structure
        chunks: list[Chunk] = []
        current_part = ""
        current_chapter = ""

        for marker in markers:
            if marker["type"] == "part":
                current_part = f"Part {marker['number']}: {marker['title']}".strip()
                continue
            elif marker["type"] == "chapter":
                current_chapter = f"Chapter {marker['number']}: {marker['title']}".strip()
                continue

            section_text = text[marker["start_pos"]:marker["end_pos"]].strip()
            section_tokens = count_tokens(section_text)

            base_metadata = ChunkMetadata(
                act_name=act_name,
                act_id=act_id,
                part=current_part,
                chapter=current_chapter,
                section_number=marker.get("number", ""),
                section_title=marker.get("title", ""),
                category=category,
                year=year,
                language=language,
            )

            if section_tokens <= self.chunk_size_tokens:
                if section_tokens >= self.min_section_tokens:
                    chunks.append(self._create_chunk(
                        text=section_text,
                        metadata=base_metadata,
                        chunk_index=len(chunks),
                    ))
            else:
                chunks.extend(
                    self._split_large_section(section_text, base_metadata, len(chunks))
                )

        # Step 3: Handle any text before the first marker (preamble)
        if markers and markers[0]["start_pos"] > 100:
            preamble = text[: markers[0]["start_pos"]].strip()
            preamble_tokens = count_tokens(preamble)
            if preamble_tokens >= self.min_chunk_tokens:
                preamble_metadata = ChunkMetadata(
                    act_name=act_name,
                    act_id=act_id,
                    section_number="preamble",
                    section_title="Preamble / Preliminary",
                    category=category,
                    year=year,
                    language=language,
                )
                if preamble_tokens > self.chunk_size_tokens:
                    preamble_chunks = self._split_large_section(
                        preamble, preamble_metadata, 0
                    )
                else:
                    preamble_chunks = [
                        self._create_chunk(preamble, preamble_metadata, 0)
                    ]
                chunks = preamble_chunks + chunks
                for i, chunk in enumerate(chunks):
                    chunk.chunk_index = i

        logger.info(
            f"Created {len(chunks)} chunks from '{act_name}' "
            f"(avg {sum(c.token_count for c in chunks) // max(len(chunks), 1)} tokens/chunk)"
        )

        return chunks

    def _split_large_section(
        self,
        text: str,
        base_metadata: ChunkMetadata,
        start_index: int,
    ) -> list[Chunk]:
        """
        Split a large section into smaller chunks.

        Order of preference: paragraphs -> sentences -> hard token slices.
        Guarantees no returned chunk exceeds the embedding token limit.
        """
        units = self._to_units(text)

        chunks: list[Chunk] = []
        current_text = ""
        current_tokens = 0
        separator = "\n\n" if "\n\n" in text else " "

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            unit_tokens = count_tokens(unit)

            if current_tokens + unit_tokens > self.chunk_size_tokens and current_text:
                chunks.append(self._create_chunk(
                    text=current_text.strip(),
                    metadata=base_metadata,
                    chunk_index=start_index + len(chunks),
                ))
                overlap_text = self._get_overlap_text(current_text)
                current_text = (
                    overlap_text + separator + unit if overlap_text else unit
                )
                current_tokens = count_tokens(current_text)
            else:
                current_text += (separator + unit if current_text else unit)
                current_tokens += unit_tokens

        if current_text.strip() and count_tokens(current_text) >= self.min_chunk_tokens:
            chunks.append(self._create_chunk(
                text=current_text.strip(),
                metadata=base_metadata,
                chunk_index=start_index + len(chunks),
            ))

        for offset, chunk in enumerate(chunks):
            chunk.chunk_index = start_index + offset

        return chunks

    def _to_units(self, text: str) -> list[str]:
        """
        Break text into units each guaranteed to fit the token budget.

        Paragraphs first; any paragraph still too large is split into sentences;
        any sentence still too large is hard-sliced by token count. This is the
        safety net that prevents oversized chunks reaching the embedding API.
        """
        units: list[str] = []

        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue

            if count_tokens(para) <= self.chunk_size_tokens:
                units.append(para)
                continue

            for sentence in re.split(r"(?<=[.!?])\s+", para):
                sentence = sentence.strip()
                if not sentence:
                    continue

                if count_tokens(sentence) <= self.chunk_size_tokens:
                    units.append(sentence)
                else:
                    units.extend(slice_by_tokens(sentence, self.chunk_size_tokens))

        return units

    def _fallback_chunk(
        self,
        text: str,
        act_name: str,
        act_id: str,
        category: str,
        year: int,
        language: str,
    ) -> list[Chunk]:
        """Fallback chunking when no structural markers are found."""
        logger.info("Using fallback paragraph-based chunking")

        base_metadata = ChunkMetadata(
            act_name=act_name,
            act_id=act_id,
            category=category,
            year=year,
            language=language,
        )

        return self._split_large_section(text, base_metadata, 0)

    def _get_overlap_text(self, text: str) -> str:
        """Extract the trailing sentences of text for overlap continuity."""
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)

        overlap_text = ""
        overlap_tokens = 0

        for sentence in reversed(sentences):
            sent_tokens = count_tokens(sentence)
            if overlap_tokens + sent_tokens > self.chunk_overlap_tokens:
                break
            overlap_text = sentence + " " + overlap_text
            overlap_tokens += sent_tokens

        return overlap_text.strip()

    def _create_chunk(
        self, text: str, metadata: ChunkMetadata, chunk_index: int
    ) -> Chunk:
        """Create a Chunk object with a unique, deterministic ID."""
        hash_input = f"{metadata.act_id}:{chunk_index}:{text[:100]}"
        chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        return Chunk(
            chunk_id=f"{metadata.act_id}_{chunk_id}",
            text=text,
            token_count=count_tokens(text),
            metadata=metadata,
            chunk_index=chunk_index,
        )