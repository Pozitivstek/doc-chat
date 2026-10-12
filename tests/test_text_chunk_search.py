"""Нормализация текста, нарезка на фрагменты и поиск BM25 (русский и английский)."""
import unittest

from docchat.chunking import Chunk, chunk_pages, clean_text, split_page
from docchat.search import Bm25Index
from docchat.text import tokenize


class TokenizeTest(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])

    def test_russian_inflections_share_a_stem(self):
        forms = ["документ", "документы", "документов", "документе", "документами"]
        self.assertEqual(len({tokenize(f)[0] for f in forms}), 1)

    def test_english_plurals_and_suffixes(self):
        self.assertEqual(tokenize("contracts")[0], tokenize("contract")[0])
        self.assertEqual(tokenize("payments")[0], tokenize("payment")[0])

    def test_yo_is_normalised(self):
        self.assertEqual(tokenize("ёлка"), tokenize("елка"))

    def test_stopwords_are_dropped(self):
        self.assertEqual(tokenize("the и в on"), [])

    def test_numbers_are_kept_verbatim(self):
        self.assertIn("2026", tokenize("срок до 2026 года"))

    def test_short_tokens_are_dropped(self):
        self.assertEqual(tokenize("а б в x"), [])

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class CleanTextTest(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(clean_text("a   b\t\tc"), "a b c")

    def test_joins_hyphenated_line_breaks(self):
        self.assertEqual(clean_text("соглаше-\nние сторон"), "соглашение сторон")

    def test_single_newlines_become_spaces_paragraphs_stay(self):
        self.assertEqual(clean_text("строка один\nстрока два\n\nновый абзац"),
                         "строка один строка два\n\nновый абзац")

    def test_removes_control_characters(self):
        self.assertEqual(clean_text("a\x00b\x07c"), "abc")


class SplitPageTest(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_page("Короткий текст."), ["Короткий текст."])

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(split_page("   \n  "), [])

    def test_long_text_is_split_below_hard_max(self):
        sentence = "Это довольно длинное предложение для проверки нарезки. "
        text = sentence * 80
        parts = split_page(text, target=500, hard_max=700)
        self.assertGreater(len(parts), 3)
        self.assertTrue(all(len(p) <= 700 for p in parts))

    def test_consecutive_chunks_overlap_by_a_sentence(self):
        sentences = [f"Предложение номер {i} содержит достаточно слов для теста." for i in range(40)]
        parts = split_page(" ".join(sentences), target=300, hard_max=500)
        for before, after in zip(parts, parts[1:]):
            last_sentence = before.rsplit(". ", 1)[-1] if ". " in before else before
            self.assertIn(last_sentence.strip().rstrip("."), after)

    def test_unbroken_giant_token_is_hard_cut(self):
        parts = split_page("x" * 5000, target=500, hard_max=700)
        self.assertTrue(all(len(p) <= 700 for p in parts))
        self.assertEqual("".join(parts), "x" * 5000)


class ChunkPagesTest(unittest.TestCase):
    def test_chunks_keep_page_numbers_and_global_order(self):
        chunks = chunk_pages("doc1", ["Первая страница.", "", "Третья страница."])
        self.assertEqual([(c.page, c.text) for c in chunks],
                         [(1, "Первая страница."), (3, "Третья страница.")])
        self.assertEqual([c.index for c in chunks], [0, 1])
        self.assertTrue(all(c.doc_id == "doc1" for c in chunks))


CORPUS = [
    Chunk("d1", 0, 1, "Срок действия договора составляет двенадцать месяцев с даты подписания."),
    Chunk("d1", 1, 2, "Оплата производится в течение десяти рабочих дней после выставления счёта."),
    Chunk("d1", 2, 3, "Стороны несут ответственность в соответствии с законодательством."),
    Chunk("d2", 0, 1, "The warranty period is twelve months from the delivery date."),
    Chunk("d2", 1, 2, "Payment terms: invoices are due within thirty days."),
]


class Bm25Test(unittest.TestCase):
    def setUp(self):
        self.index = Bm25Index(CORPUS)

    def test_finds_russian_passage_despite_inflection(self):
        hits = self.index.search("какой срок у договора?")
        self.assertEqual(hits[0].chunk.page, 1)
        self.assertEqual(hits[0].chunk.doc_id, "d1")

    def test_finds_english_passage(self):
        hits = self.index.search("when is payment due")
        self.assertEqual(hits[0].chunk.doc_id, "d2")
        self.assertEqual(hits[0].chunk.page, 2)

    def test_results_are_sorted_and_limited(self):
        hits = self.index.search("оплата договора срок", k=2)
        self.assertLessEqual(len(hits), 2)
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))

    def test_no_overlap_returns_nothing(self):
        self.assertEqual(self.index.search("квантовая хромодинамика"), [])

    def test_stopword_only_query_returns_nothing(self):
        self.assertEqual(self.index.search("и в на"), [])

    def test_empty_index(self):
        self.assertEqual(Bm25Index([]).search("что угодно"), [])

    def test_weak_hits_are_dropped_relative_to_the_best(self):
        corpus = [Chunk("d", 0, 1, "Гарантия на кофе: срок хранения составляет шесть месяцев с даты обжарки.")]
        corpus += [Chunk("d", i, i + 1, f"Пункт {i}. Договор заключён между сторонами и вступает в силу.")
                   for i in range(1, 6)]
        hits = Bm25Index(corpus).search("какая гарантия на кофе по договору")
        self.assertEqual([h.chunk.page for h in hits], [1])

    def test_best_hit_is_always_kept(self):
        hits = Bm25Index(CORPUS).search("срок", k=6)
        self.assertGreaterEqual(len(hits), 1)

    def test_rare_term_outweighs_common_term(self):
        corpus = [Chunk("d", i, 1, "общий общий общий текст договор") for i in range(5)]
        corpus.append(Chunk("d", 5, 1, "уникальный термин сингулярность"))
        hits = Bm25Index(corpus).search("договор сингулярность")
        self.assertEqual(hits[0].chunk.index, 5)


if __name__ == "__main__":
    unittest.main()
