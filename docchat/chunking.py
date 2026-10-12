"""Очистка текста страницы и нарезка на фрагменты с перекрытием в одно предложение."""
import re
from dataclasses import dataclass

TARGET = 800      # желаемый размер фрагмента, символов
HARD_MAX = 1200   # больше не бывает
OVERLAP_MAX = TARGET // 2

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int      # сквозной номер в документе
    page: int       # 1-based
    text: str


def clean_text(raw: str) -> str:
    text = _CONTROL.sub("", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    paragraphs = (re.sub(r"\s+", " ", p).strip() for p in _PARAGRAPH.split(text))
    return "\n\n".join(p for p in paragraphs if p)


def _units(text: str, hard_max: int) -> list[str]:
    """Предложения; слишком длинные куски режутся жёстко."""
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        for sentence in _SENTENCE.split(paragraph):
            sentence = sentence.strip()
            while len(sentence) > hard_max:
                units.append(sentence[:hard_max])
                sentence = sentence[hard_max:]
            if sentence:
                units.append(sentence)
    return units


def split_page(text: str, target: int = TARGET, hard_max: int = HARD_MAX) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= target:
        return [cleaned]

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in _units(cleaned, hard_max):
        if current and length + 1 + len(unit) > target:
            chunks.append(" ".join(current))
            tail = current[-1]
            keep_overlap = len(tail) <= target // 2 and len(tail) + 1 + len(unit) <= hard_max
            current = [tail] if keep_overlap else []
            length = len(tail) if keep_overlap else 0
        current.append(unit)
        length += len(unit) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(doc_id: str, pages: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for number, page_text in enumerate(pages, start=1):
        for piece in split_page(page_text):
            chunks.append(Chunk(doc_id, len(chunks), number, piece))
    return chunks
