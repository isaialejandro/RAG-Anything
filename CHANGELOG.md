# Changelog

All notable changes to this fork are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This fork tracks [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything). Entries
below describe changes made on top of upstream; upstream releases are not restated.

## [Unreleased]

Python 3.14 support, verified on macOS x86_64 against upstream 1.4.5.

### Context

Upstream already excludes MinerU on Python 3.14 (`mineru[core]...; python_version < "3.14"`),
which leaves Docling as the only parser there — but nothing in the packaging declared how
to install Docling. Separately, the full Docling distribution cannot be installed on Intel
macOS at all: it depends on `docling-ibm-models`, which requires PyTorch, and PyTorch has
published no macOS `x86_64` wheel since 2.2.2. Both gaps are addressed here.

The project's own code needed almost nothing: the test suite passes on 3.14 once the
environment respects the constraints the project already declares.

### Added

- `docling` extra that installs the Docling parser, selecting `docling-slim` on Intel
  macOS (where the full distribution cannot resolve) and `docling` on every other
  platform. `docling-slim` exposes the same `DocumentConverter` and `PdfFormatOption`
  API that `raganything/parser.py` imports, without the TableFormer layout models.
- Python 3.13 and 3.14 in the CI test matrix.
- CI step asserting that the `[all]` extra leaves an importable parser, so the gap fixed
  below cannot silently return.
- `CHANGELOG.md` (this file).
- `examples/raganything_claude.py`: end-to-end example pairing Claude as the text and
  vision model with Ollama `bge-m3` embeddings, including the OpenAI-to-Anthropic
  message conversion needed for VLM-enhanced queries. Reads all credentials and hosts
  from the environment.

### Fixed

- `[all]` left no usable parser on Python 3.14: MinerU is skipped there by an upstream
  marker, and Docling was never part of the extra. `[all]` now includes Docling.
- `paddleocr` is skipped on Python 3.14, where `paddlepaddle` publishes no cp314 wheel
  and would fail the whole `[all]` resolution.
- `BatchParser.process_batch_async` called the deprecated `asyncio.get_event_loop()`
  from inside a coroutine; it now uses `asyncio.get_running_loop()`.
- Docling installation hints in parser errors and logs pointed at a bare
  `pip install docling`, which cannot resolve on Intel macOS. They now name the
  `raganything[docling]` extra, which resolves on every supported platform.
- `tests/test_chinese_cid_font.py` failed instead of skipping when the optional
  `reportlab` dependency was absent.
- `tests/test_enhanced_markdown_css.py` failed when the optional `markdown` dependency
  was absent, even though the test supplies its own fake module. It no longer requires
  the real package to be importable.

### Changed

- README (English and Chinese), `docs/batch_processing.md` and `requirements.txt`
  corrected: they stated that Docling was not included in `[all]` and recommended a bare
  `pip install docling`. Both statements are now false and were updated, with the Intel
  macOS constraint documented.

### Known limitations

- **Intel macOS has no TableFormer.** `docling-slim` omits the ML layout models, so
  complex tables (merged cells, multi-level headers, borderless) parse with lower
  fidelity. Text PDFs, Office formats, Markdown, HTML and images are unaffected.
- **MinerU and PaddleOCR are unavailable on Python 3.14**, by upstream constraint and
  missing cp314 wheels respectively. Use Python 3.13 or earlier if either is required.
- `lightrag-hku<1.5` is a permanent upstream constraint, not a temporary pin: RAG-Anything
  was merged into LightRAG 1.5+, and this line's multimodal extraction is incompatible
  with it. Installing `lightrag-hku>=1.5` breaks `raganything/processor.py`, which imports
  `chunking_by_token_size` from `lightrag.operate`.
