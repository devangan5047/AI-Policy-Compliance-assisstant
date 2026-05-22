from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class PDFService:
    def extract_text(self, file_path: str | Path) -> str:
        reader = PdfReader(Path(file_path))
        return self._extract_pages(reader)

    def extract_text_from_bytes(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        return self._extract_pages(reader)

    def _extract_pages(self, reader: PdfReader) -> str:
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"\n[Page {page_number}]\n{text}")
        return "\n".join(pages)
