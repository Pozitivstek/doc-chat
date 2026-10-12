"""Чтение PDF/текста, хранилище с изоляцией сессий, лимитер запросов."""
import io
import unittest
from pathlib import Path

from pypdf import PdfWriter

from docchat.pdf import DocumentError, extract_pages, extract_text_file
from docchat.ratelimit import RateLimiter
from docchat.store import MAX_DOCS_PER_SESSION, LimitError, Store
from tests.helpers import make_pdf


class PdfTest(unittest.TestCase):
    def test_extracts_text_per_page(self):
        pages = extract_pages(make_pdf(["Hello first page", "Second page here"]))
        self.assertEqual(len(pages), 2)
        self.assertIn("Hello first page", pages[0])
        self.assertIn("Second page here", pages[1])

    def test_rejects_non_pdf(self):
        with self.assertRaises(DocumentError) as ctx:
            extract_pages(b"MZ\x90\x00 this is an exe")
        self.assertEqual(ctx.exception.code, "not_pdf")

    def test_rejects_truncated_pdf(self):
        with self.assertRaises(DocumentError) as ctx:
            extract_pages(make_pdf(["abc"])[:80])
        self.assertIn(ctx.exception.code, {"corrupt", "no_text"})

    def test_scanned_pdf_without_text_layer(self):
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaises(DocumentError) as ctx:
            extract_pages(buffer.getvalue())
        self.assertEqual(ctx.exception.code, "no_text")

    def test_page_limit(self):
        with self.assertRaises(DocumentError) as ctx:
            extract_pages(make_pdf(["page text"] * 5), max_pages=3)
        self.assertEqual(ctx.exception.code, "too_many_pages")

    def test_encrypted_pdf_is_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret")
        buffer = io.BytesIO()
        writer.write(buffer)
        with self.assertRaises(DocumentError) as ctx:
            extract_pages(buffer.getvalue())
        self.assertEqual(ctx.exception.code, "encrypted")

    def test_russian_pdf_roundtrip(self):
        from fpdf import FPDF

        font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
        if not font.exists():
            self.skipTest("нет системного шрифта с кириллицей")
        pdf = FPDF()
        pdf.add_font("Arial", fname=str(font))
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, "Срок действия договора составляет двенадцать месяцев.")
        pages = extract_pages(bytes(pdf.output()))
        self.assertIn("двенадцать месяцев", pages[0])


class TextFileTest(unittest.TestCase):
    def test_utf8_text(self):
        pages = extract_text_file("Привет, мир! ".encode() * 10)
        self.assertEqual(len(pages), 1)
        self.assertIn("Привет", pages[0])

    def test_cp1251_fallback(self):
        pages = extract_text_file("Договор поставки".encode("cp1251"))
        self.assertIn("Договор", pages[0])

    def test_long_text_is_paginated(self):
        pages = extract_text_file(("слово " * 2000).encode())
        self.assertGreater(len(pages), 1)

    def test_binary_garbage_is_rejected(self):
        with self.assertRaises(DocumentError):
            extract_text_file(bytes(range(256)) * 20)

    def test_empty_text_is_rejected(self):
        with self.assertRaises(DocumentError) as ctx:
            extract_text_file(b"   \n ")
        self.assertEqual(ctx.exception.code, "no_text")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def add(self, sid="s1", name="a.pdf", pages=("Первая страница документа.", "Вторая страница.")):
        return self.store.add_document(sid, name, list(pages))

    def test_add_and_list(self):
        meta = self.add()
        self.assertEqual((meta.name, meta.pages), ("a.pdf", 2))
        self.assertGreaterEqual(meta.chunks, 2)
        self.assertEqual([d.id for d in self.store.list_documents("s1")], [meta.id])

    def test_sessions_are_isolated(self):
        meta = self.add("s1")
        self.assertEqual(self.store.list_documents("s2"), [])
        self.assertFalse(self.store.delete_document("s2", meta.id))
        self.assertEqual(len(self.store.list_documents("s1")), 1)

    def test_delete_removes_chunks(self):
        meta = self.add()
        self.assertTrue(self.store.delete_document("s1", meta.id))
        self.assertEqual(self.store.chunks_for("s1", None), [])
        self.assertFalse(self.store.delete_document("s1", meta.id))

    def test_chunks_for_selected_documents_only(self):
        first = self.add(name="a.pdf", pages=("Alpha text",))
        second = self.add(name="b.pdf", pages=("Beta text",))
        chunks = self.store.chunks_for("s1", [second.id])
        self.assertEqual({c.doc_id for c in chunks}, {second.id})
        self.assertEqual({c.doc_id for c in self.store.chunks_for("s1", None)}, {first.id, second.id})

    def test_foreign_document_ids_are_ignored(self):
        mine = self.add("s1")
        theirs = self.add("s2")
        chunks = self.store.chunks_for("s1", [mine.id, theirs.id])
        self.assertEqual({c.doc_id for c in chunks}, {mine.id})

    def test_document_limit(self):
        for i in range(MAX_DOCS_PER_SESSION):
            self.add(name=f"{i}.pdf")
        with self.assertRaises(LimitError):
            self.add(name="extra.pdf")

    def test_names_of_documents(self):
        meta = self.add(name="договор.pdf")
        self.assertEqual(self.store.document_names("s1"), {meta.id: "договор.pdf"})

    def test_purge_old_documents(self):
        self.add()
        self.assertEqual(self.store.purge_older_than(0, now_offset=10), 1)
        self.assertEqual(self.store.list_documents("s1"), [])

    def test_fresh_documents_survive_purge(self):
        self.add()
        self.assertEqual(self.store.purge_older_than(3600), 0)


class RateLimiterTest(unittest.TestCase):
    def test_blocks_after_limit_and_recovers(self):
        now = [0.0]
        limiter = RateLimiter(max_events=3, window=60, clock=lambda: now[0])
        self.assertEqual([limiter.allow("a") for _ in range(4)], [True, True, True, False])
        now[0] = 61
        self.assertTrue(limiter.allow("a"))

    def test_keys_are_independent(self):
        limiter = RateLimiter(max_events=1, window=60, clock=lambda: 0.0)
        self.assertTrue(limiter.allow("a"))
        self.assertTrue(limiter.allow("b"))
        self.assertFalse(limiter.allow("a"))


if __name__ == "__main__":
    unittest.main()
