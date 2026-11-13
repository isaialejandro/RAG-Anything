# RAG-Anything on Linux with the full Docling parser.
#
# Why this image exists: on Intel macOS + CPython 3.14 there is no installable
# PyTorch wheel, so the Docling PDF pipeline (layout + tables) cannot run on the
# host at all. On linux/amd64, PyTorch *does* publish cp314 wheels, so the same
# Python version the host runs gets the full, torch-backed parser here instead —
# without touching any package on the Mac.
FROM python:3.14-slim-trixie

# System libraries Docling's full pipeline needs at runtime:
#   libgl1 / libglib2.0-0 — OpenCV, pulled transitively by the layout models.
# build-essential is kept so any dependency without a cp314 wheel can still
# compile from sdist rather than failing the build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Docling downloads its layout/TableFormer models here on first run; the
    # compose file mounts a named volume so they persist across containers.
    HF_HOME=/models

WORKDIR /app

# Install dependencies first, against the packaging metadata only, so the heavy
# torch + Docling layer is cached and not rebuilt on every source change.
COPY pyproject.toml MANIFEST.in README.md requirements.txt ./
COPY raganything/__init__.py raganything/__init__.py
RUN pip install ".[docling,image,text]" anthropic python-dotenv

# Now copy the rest of the source. Editable so local edits (bind-mounted in
# compose) take effect without a rebuild.
COPY . .
RUN pip install --no-deps -e .

# Fail the build if the parser or the LightRAG pin regressed, rather than at
# ingest time inside a running container.
RUN python - <<'PY'
import importlib.util, sys
assert importlib.util.find_spec("docling"), "docling parser not installed"
assert importlib.util.find_spec("torch"), "torch not installed (layout pipeline needs it)"
from lightrag.operate import chunking_by_token_size  # noqa: F401  (lightrag-hku<1.5)
print(f"OK python {sys.version_info.major}.{sys.version_info.minor}: docling + torch present")
PY

ENTRYPOINT ["python", "examples/raganything_claude.py"]
CMD ["--help"]
