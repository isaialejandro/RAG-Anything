"""
End-to-end smoke test: parse a real text PDF through the full Docling pipeline.

Opt-in because it needs torch and downloads Docling's layout models on first
run. It is the automated counterpart to running the Docker image by hand, and
is meant to run inside that image (linux/amd64, torch present):

    RUN_DOCLING_PDF_SMOKE=1 pytest tests/test_docling_pdf_smoke.py

On a torch-less host (e.g. Intel macOS) it skips, since Docling's PDF pipeline
cannot load its layout model there.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCLING_PDF_SMOKE") != "1",
    reason="opt-in: set RUN_DOCLING_PDF_SMOKE=1 (needs torch + Docling models)",
)


def _write_text_pdf(path: Path, lines):
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()


def test_docling_parses_text_pdf(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("docling")
    from raganything.parser import DoclingParser

    pdf = tmp_path / "sample.pdf"
    _write_text_pdf(
        pdf,
        [
            "Essential Skills for Agile Development",
            "This document describes iterative delivery and refactoring.",
            "Continuous integration keeps the build green.",
        ],
    )

    blocks = DoclingParser().parse_pdf(
        str(pdf), output_dir=str(tmp_path / "out"), method="auto"
    )

    assert blocks, "Docling returned no content blocks"
    text = " ".join(
        b.get("text", "") for b in blocks if b.get("type") == "text"
    ).lower()
    assert "agile development" in text
    assert "continuous integration" in text
