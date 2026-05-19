from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from pypdf import PdfReader


def extract_pdf_text(file: BinaryIO | BytesIO) -> str:
    """Extract text from every readable page in an uploaded PDF."""
    reader = PdfReader(file)
    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages)

