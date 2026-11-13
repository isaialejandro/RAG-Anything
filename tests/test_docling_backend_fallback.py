"""
Tests for DoclingParser's graceful degradation when optional ML models are
absent (docling-slim / torch-less platforms such as CPython 3.14 on Intel
macOS).

These tests do not parse any document; they verify the decision logic that
picks a PDF backend and disables table/OCR features, so they run fast and
everywhere, independently of whether torch or the Docling models are installed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from raganything.parser import DoclingParser  # noqa: E402


def _blocking_find_spec(blocked):
    """Return a find_spec that reports `blocked` names as missing."""
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name in blocked:
            return None
        return real(name, *args, **kwargs)

    return fake


def test_model_availability_reports_all_missing(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        _blocking_find_spec({"docling_ibm_models", "easyocr", "docling_parse"}),
    )
    assert DoclingParser._docling_model_availability() == {
        "tableformer": False,
        "easyocr": False,
        "docling_parse": False,
    }


def test_model_availability_reports_all_present(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: sentinel)
    assert DoclingParser._docling_model_availability() == {
        "tableformer": True,
        "easyocr": True,
        "docling_parse": True,
    }


def _patch_docling_converter(monkeypatch):
    """Replace Docling's converter classes with recorders; return the capture."""
    dc = pytest.importorskip("docling.document_converter")
    captured = {}

    class FakePdfFormatOption:
        def __init__(self, **kwargs):
            captured["pdf_kwargs"] = kwargs

    class FakeDocumentConverter:
        def __init__(self, format_options=None):
            captured["format_options"] = format_options

    monkeypatch.setattr(dc, "PdfFormatOption", FakePdfFormatOption)
    monkeypatch.setattr(dc, "DocumentConverter", FakeDocumentConverter)
    return captured


def test_converter_disables_tables_and_ocr_when_models_absent(monkeypatch):
    pytest.importorskip("docling")
    captured = _patch_docling_converter(monkeypatch)
    monkeypatch.setattr(
        DoclingParser,
        "_docling_model_availability",
        staticmethod(
            lambda: {"tableformer": False, "easyocr": False, "docling_parse": True}
        ),
    )

    DoclingParser()._get_converter(tables=True, allow_ocr=True)

    opts = captured["pdf_kwargs"]["pipeline_options"]
    assert opts.do_table_structure is False
    assert opts.do_ocr is False


def test_converter_forces_pypdfium_backend_when_docling_parse_absent(monkeypatch):
    pytest.importorskip("docling")
    PyPdfium = pytest.importorskip(
        "docling.backend.pypdfium2_backend"
    ).PyPdfiumDocumentBackend
    captured = _patch_docling_converter(monkeypatch)
    monkeypatch.setattr(
        DoclingParser,
        "_docling_model_availability",
        staticmethod(
            lambda: {
                "tableformer": False,
                "easyocr": False,
                "docling_parse": False,
            }
        ),
    )

    DoclingParser()._get_converter(tables=True, allow_ocr=True)

    assert captured["pdf_kwargs"].get("backend") is PyPdfium


def test_converter_keeps_default_backend_when_docling_parse_present(monkeypatch):
    pytest.importorskip("docling")
    captured = _patch_docling_converter(monkeypatch)
    monkeypatch.setattr(
        DoclingParser,
        "_docling_model_availability",
        staticmethod(
            lambda: {"tableformer": True, "easyocr": True, "docling_parse": True}
        ),
    )

    DoclingParser()._get_converter(tables=True, allow_ocr=True)

    # No backend override: Docling keeps its default (docling-parse).
    assert "backend" not in captured["pdf_kwargs"]
    opts = captured["pdf_kwargs"]["pipeline_options"]
    assert opts.do_table_structure is True
    assert opts.do_ocr is True
