# Running RAG-Anything in Docker (full Docling parser)

## Why

On **Intel macOS** (and any `x86_64` macOS), PyTorch publishes no wheel for
CPython 3.14 — the last macOS `x86_64` build was 2.2.2 (cp38–cp312). Docling's
PDF pipeline always loads a torch-based layout model, so **PDFs and images
cannot be parsed on that host**, at any Python version. Office, Markdown and
HTML still parse there through `docling-slim`, but PDFs do not.

On **`linux/amd64`, PyTorch does ship cp314 wheels.** This image runs the same
CPython 3.14 the host runs, with the full, torch-backed Docling parser — so PDF
layout, tables and images work in the container while **nothing is installed on
the Mac**. An Intel Mac runs this image natively (no emulation).

## Prerequisites

- Docker (Colima, Docker Desktop, or any `linux/amd64` engine).
- A reachable Ollama host serving the embedding model (this project uses
  `bge-m3` on an intranet host).
- A `.env` file at the repo root (next to `docker-compose.yml`) with:

  ```env
  ANTHROPIC_API_KEY=sk-ant-...
  EMBEDDING_BINDING=ollama
  EMBEDDING_BINDING_HOST=http://10.0.0.111:11434
  EMBEDDING_MODEL=bge-m3:latest
  EMBEDDING_DIM=1024
  ```

  The `.env` is **never** copied into the image (see `.dockerignore`); Compose
  injects it at runtime via `env_file`. Rotate any key that has been shared in
  chat or logs.

  `EMBEDDING_BINDING_HOST` is reached from inside the container: for a remote
  Ollama on the LAN use its address (e.g. `http://10.0.0.111:11434`); if Ollama
  runs on the Mac host instead, use `http://host.docker.internal:11434`, not
  `localhost` (which would point at the container itself).

## Build

```bash
docker compose build
# or: docker build -t raganything:py314 .
```

The build installs `raganything[docling,image,text]` plus `anthropic`. On
`linux/amd64` the `docling` extra resolves to the **full** Docling distribution
(torch-backed), unlike the `docling-slim` fallback used on Intel macOS. A final
build step asserts that `docling`, `torch` and the `lightrag-hku<1.5` API are all
importable, so a regression fails the build instead of the first ingest.

## Run

Put input documents under `./data` (bind-mounted read-only at `/data`), then:

```bash
docker compose run --rm rag /data/your-document.pdf
```

Outputs persist on the host via bind mounts:

| Host path       | Container path     | Contents                                   |
|-----------------|--------------------|--------------------------------------------|
| `./rag_storage` | `/app/rag_storage` | Knowledge graph + vector stores (LightRAG) |
| `./output`      | `/app/output`      | Parsed artifacts (Docling JSON/Markdown)   |
| `docling-models` (named volume) | `/models` | Docling layout/TableFormer models (downloaded once) |

The first run downloads Docling's models into the `docling-models` volume; later
runs reuse them. The container reaches the remote Ollama host over the LAN using
`EMBEDDING_BINDING_HOST` from `.env`.

## What a run does

`examples/raganything_claude.py` (the image entrypoint):

1. **Parse** the document with Docling → a list of content blocks (text, tables,
   images, equations).
2. **Chunk + embed** each block via Ollama `bge-m3` (1024-dim).
3. **Extract** entities and relations with Claude, building a knowledge graph in
   `rag_storage`.
4. **Query** the graph (the script runs two sample queries and logs the answers;
   logs also go to `raganything_claude.log`).

## Smoke test inside the image

The runtime image is lean and does not ship pytest; install it on the fly:

```bash
docker compose run --rm --entrypoint bash rag -lc \
  "pip install -q pytest pytest-asyncio && \
   RUN_DOCLING_PDF_SMOKE=1 pytest tests/test_docling_pdf_smoke.py -v"
```

This parses a generated text PDF through the full pipeline and asserts the text
was extracted. It is skipped on torch-less hosts.

## Notes

- **MinerU** is intentionally not installed: it requires Python ≤3.13, and the
  full Docling parser already covers layout, tables and images here.
- The image is a one-shot runner, not a server; there is nothing to keep running.
- To parse non-PDF formats on the Mac host directly (no Docker), `docling-slim`
  handles Office, Markdown and HTML on CPython 3.14.
