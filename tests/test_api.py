"""HTTP API: загрузка, изоляция сессий, лимиты, чат со стримингом (SSE)."""
import json
import unittest

from fastapi.testclient import TestClient

from docchat.app import create_app
from docchat.config import Settings
from docchat.llm import AnswerError, Delta, Done
from docchat.store import Store
from tests.helpers import make_pdf, parse_sse

CONTRACT = make_pdf([
    "The contract term is twelve months from the signing date.",
    "Payment is due within ten business days after the invoice is issued.",
])


class RecordingAnswerer:
    """Запоминает, что получил, и отвечает фиксированным текстом."""

    def __init__(self, parts=("Срок — двенадцать месяцев ", "[1]."), error=None):
        self.parts = parts
        self.error = error
        self.calls = []

    async def stream_answer(self, question, history, sources):
        self.calls.append({"question": question, "history": history, "sources": sources})
        for part in self.parts:
            yield Delta(part)
        if self.error:
            raise self.error
        yield Done(usage={"input_tokens": 100, "output_tokens": 10}, model="fake")


def build(**settings):
    answerer = settings.pop("answerer", None) or RecordingAnswerer()
    config = Settings(db_path=":memory:", **settings)
    store = Store(":memory:")
    app = create_app(config, store=store, answerer=answerer)
    return TestClient(app), answerer, store


def upload(client, data=CONTRACT, name="contract.pdf", mime="application/pdf"):
    return client.post("/api/docs", files={"file": (name, data, mime)})


def chat(client, question, **body):
    return client.post("/api/chat", json={"question": question, **body})


class UploadTest(unittest.TestCase):
    def test_upload_returns_metadata_and_sets_cookie(self):
        client, _, _ = build()
        response = upload(client)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["name"], body["pages"]), ("contract.pdf", 2))
        self.assertTrue(body["id"])
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)

    def test_list_and_delete(self):
        client, _, _ = build()
        doc_id = upload(client).json()["id"]
        self.assertEqual([d["id"] for d in client.get("/api/docs").json()["documents"]], [doc_id])
        self.assertEqual(client.delete(f"/api/docs/{doc_id}").status_code, 200)
        self.assertEqual(client.get("/api/docs").json()["documents"], [])
        self.assertEqual(client.delete(f"/api/docs/{doc_id}").status_code, 404)

    def test_sessions_do_not_see_each_others_documents(self):
        first, _, store = build()
        upload(first)
        second = TestClient(first.app)               # другой браузер → другая cookie
        self.assertEqual(second.get("/api/docs").json()["documents"], [])
        doc_id = first.get("/api/docs").json()["documents"][0]["id"]
        self.assertEqual(second.delete(f"/api/docs/{doc_id}").status_code, 404)

    def test_text_file_is_accepted(self):
        client, _, _ = build()
        response = upload(client, "Договор поставки. Срок два года.".encode(), "note.txt", "text/plain")
        self.assertEqual(response.status_code, 200)

    def test_unsupported_type(self):
        client, _, _ = build()
        response = upload(client, b"MZ\x90\x00", "virus.exe", "application/octet-stream")
        self.assertEqual(response.status_code, 415)

    def test_pdf_extension_with_wrong_content_is_rejected(self):
        client, _, _ = build()
        response = upload(client, b"<html>not a pdf</html>", "fake.pdf")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "not_pdf")

    def test_too_large(self):
        client, _, _ = build(max_upload_bytes=1000)
        response = upload(client, CONTRACT + b"0" * 2000)
        self.assertEqual(response.status_code, 413)

    def test_too_many_pages(self):
        client, _, _ = build(max_pages=1)
        response = upload(client)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "too_many_pages")

    def test_document_limit_per_session(self):
        client, _, _ = build()
        for _ in range(5):
            self.assertEqual(upload(client).status_code, 200)
        self.assertEqual(upload(client).status_code, 409)

    def test_filename_is_sanitised(self):
        client, _, _ = build()
        body = upload(client, name="../../etc/<b>passwd</b>\x00.pdf").json()
        self.assertNotIn("/", body["name"])
        self.assertNotIn("\x00", body["name"])


class ChatTest(unittest.TestCase):
    def setUp(self):
        self.client, self.answerer, _ = build()
        self.doc_id = upload(self.client).json()["id"]

    def ask(self, question, **body):
        response = chat(self.client, question, **body)
        self.assertEqual(response.status_code, 200, response.text)
        return parse_sse(response.text)

    def test_sources_then_deltas_then_done(self):
        events = self.ask("What is the contract term?")
        names = [name for name, _ in events]
        self.assertEqual(names[0], "sources")
        self.assertEqual(names[-1], "done")
        self.assertEqual(names.count("delta"), 2)
        sources = events[0][1]["sources"]
        self.assertEqual(sources[0]["n"], 1)
        self.assertEqual(sources[0]["page"], 1)
        self.assertEqual(sources[0]["doc_name"], "contract.pdf")
        self.assertIn("twelve months", sources[0]["text"])
        self.assertEqual("".join(d["text"] for n, d in events if n == "delta"),
                         "Срок — двенадцать месяцев [1].")

    def test_answerer_receives_question_and_sources(self):
        self.ask("What is the contract term?")
        call = self.answerer.calls[0]
        self.assertEqual(call["question"], "What is the contract term?")
        self.assertEqual(call["sources"][0].page, 1)

    def test_nothing_found_does_not_call_the_model(self):
        events = self.ask("quantum chromodynamics lagrangian")
        self.assertEqual(self.answerer.calls, [])
        self.assertEqual(events[0][1]["sources"], [])
        text = "".join(d["text"] for n, d in events if n == "delta")
        self.assertIn("не нашлось", text)
        self.assertEqual(events[-1][0], "done")

    def test_follow_up_uses_previous_question_for_retrieval(self):
        history = [{"role": "user", "content": "What is the contract term?"},
                   {"role": "assistant", "content": "Twelve months."}]
        events = self.ask("And more details?", history=history)
        self.assertTrue(events[0][1]["sources"])
        self.assertEqual(self.answerer.calls[0]["history"], history)

    def test_no_documents(self):
        client, _, _ = build()
        response = chat(client, "вопрос")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "no_documents")

    def test_selected_documents_filter(self):
        other = upload(self.client, make_pdf(["Warranty lasts five years."]), "warranty.pdf").json()["id"]
        events = self.ask("warranty years", doc_ids=[other])
        self.assertEqual({s["doc_name"] for s in events[0][1]["sources"]}, {"warranty.pdf"})

    def test_foreign_document_ids_are_ignored(self):
        stranger = TestClient(self.client.app)
        foreign = upload(stranger, make_pdf(["Secret strategy of another user."]), "secret.pdf").json()["id"]
        events = self.ask("secret strategy", doc_ids=[foreign])
        self.assertNotIn("secret", json.dumps(events, ensure_ascii=False).lower())

    def test_validation(self):
        for body in ({"question": ""}, {"question": "   "}, {"question": "x" * 5000},
                     {"question": "ok", "doc_ids": "abc"}, {"question": 5}, {}):
            response = self.client.post("/api/chat", json=body)
            self.assertEqual(response.status_code, 422, body)

    def test_answerer_failure_becomes_error_event(self):
        client, _, _ = build(answerer=RecordingAnswerer(parts=("Частичный ",),
                                                         error=AnswerError("Сервис перегружен")))
        upload(client)
        events = parse_sse(chat(client, "contract term").text)
        self.assertEqual(events[-1][0], "error")
        self.assertIn("перегружен", events[-1][1]["message"])

    def test_unexpected_exception_is_not_leaked(self):
        client, _, _ = build(answerer=RecordingAnswerer(parts=(), error=RuntimeError("секретный стек")))
        upload(client)
        events = parse_sse(chat(client, "contract term").text)
        self.assertEqual(events[-1][0], "error")
        self.assertNotIn("секретный", events[-1][1]["message"])

    def test_rate_limit(self):
        client, _, _ = build(chat_rate_limit=2)
        upload(client)
        codes = [chat(client, "contract term").status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])


class MiscTest(unittest.TestCase):
    def test_config_reports_mode(self):
        client, _, _ = build()
        body = client.get("/api/config").json()
        self.assertIn(body["mode"], {"claude", "retrieval"})
        self.assertIn("max_upload_mb", body)

    def test_index_page_is_served(self):
        client, _, _ = build()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_security_headers(self):
        client, _, _ = build()
        headers = client.get("/").headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertIn("default-src 'self'", headers.get("content-security-policy", ""))


if __name__ == "__main__":
    unittest.main()
