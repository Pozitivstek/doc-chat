"""Извлечение текста из PDF и простых текстовых файлов с понятными ошибками."""
import io

from pypdf import PdfReader

from .chunking import clean_text

MIN_TEXT_CHARS = 20          # меньше — считаем, что текстового слоя нет (скан)
PSEUDO_PAGE = 3000           # размер «страницы» для .txt / .md
MAX_TEXT_CHARS = 3_000_000   # общий потолок текста одного документа


class DocumentError(Exception):
    """Документ нельзя использовать; code — машинный, message — для человека."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _unlock(reader: PdfReader) -> None:
    if not reader.is_encrypted:
        return
    try:
        opened = reader.decrypt("")   # многие «зашифрованные» PDF открываются пустым паролем
    except Exception as exc:          # нет библиотеки для AES и т. п.
        raise DocumentError("encrypted", "PDF зашифрован и не открывается без пароля.") from exc
    if not opened:
        raise DocumentError("encrypted", "PDF защищён паролем — загрузите версию без пароля.")


def extract_pages(data: bytes, *, max_pages: int = 300) -> list[str]:
    if b"%PDF-" not in data[:1024]:
        raise DocumentError("not_pdf", "Файл не похож на PDF.")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        _unlock(reader)
        count = len(reader.pages)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("corrupt", "Не получилось прочитать PDF: файл повреждён.") from exc

    if count > max_pages:
        raise DocumentError("too_many_pages", f"В PDF {count} стр., допустимо не больше {max_pages}.")

    pages: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""   # одна «битая» страница не должна ронять весь документ
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise DocumentError("too_large", "В документе слишком много текста.")
        pages.append(text)

    if sum(len(p) for p in pages) < MIN_TEXT_CHARS:
        raise DocumentError(
            "no_text", "В PDF нет текстового слоя (похоже на скан). Нужен PDF с выделяемым текстом.")
    return pages


def _decode(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise DocumentError("not_text", "Файл не похож на текстовый.")
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentError("not_text", "Не удалось определить кодировку текста.")


def _paginate(text: str, size: int = PSEUDO_PAGE) -> list[str]:
    pages, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            end = cut if cut > start + size // 2 else end
        pages.append(text[start:end].strip())
        start = end
    return [p for p in pages if p]


def extract_text_file(data: bytes) -> list[str]:
    text = clean_text(_decode(data))
    if len(text) < 1:
        raise DocumentError("no_text", "Файл пустой.")
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentError("too_large", "В файле слишком много текста.")
    return _paginate(text)
