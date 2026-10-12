"""BM25 по фрагментам: быстрый, детерминированный, без внешних моделей."""
import math
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk
from .text import tokenize

K1 = 1.5
B = 0.75
MIN_RATIO = 0.35   # фрагмент слабее 35 % от лучшего считается шумом (общие слова вроде «договор»)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Bm25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = list(chunks)
        self._tf = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df: Counter = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 6, min_ratio: float = MIN_RATIO) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(query)))   # уникальные, в порядке появления
        if not terms or not self._chunks:
            return []
        hits = []
        for position, chunk in enumerate(self._chunks):
            tf = self._tf[position]
            length = self._lengths[position] or 1
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + K1 * (1 - B + B * length / (self._avg or 1))
                score += self._idf(term) * freq * (K1 + 1) / norm
            if score > 0:
                hits.append(Hit(chunk, score))
        hits.sort(key=lambda h: (-h.score, h.chunk.doc_id, h.chunk.index))
        if not hits:
            return []
        floor = hits[0].score * min_ratio
        return [h for h in hits if h.score >= floor][:k]
