#!/usr/bin/env python
"""
RAGAnything — Claude + Ollama embeddings

LLM  : Anthropic Claude (anthropic_complete_if_cache)
Vision: Anthropic Claude with native image blocks (direct SDK call)
Embed : Ollama bge-m3:latest (local, no extra API key)

Prerequisites:
    brew install ollama          # macOS
    ollama pull bge-m3:latest
    ollama serve                 # keep running in another terminal

Usage:
    python examples/raganything_claude.py path/to/document.pdf
"""

import asyncio
import base64
import logging
import logging.config
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig
from raganything.image_signature import image_mime_type

load_dotenv(dotenv_path=".env", override=False)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OLLAMA_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
VISION_MODEL = os.getenv("VISION_MODEL", "claude-sonnet-4-5")

# Anthropic's Messages API requires max_tokens, and LightRAG's Anthropic wrapper
# does not default it — without this, entity extraction fails with
# "Missing required arguments; Expected ... 'max_tokens' ...".
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# MinerU requires Python 3.10-3.13. Default to docling on Python 3.14+.
import sys as _sys

_DEFAULT_PARSER = "docling" if _sys.version_info >= (3, 14) else "mineru"


def configure_logging():
    log_file = os.path.abspath(
        os.path.join(os.getenv("LOG_DIR", os.getcwd()), "raganything_claude.log")
    )
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "detailed",
                    "filename": log_file,
                    "maxBytes": 10_485_760,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }
    )
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """
    Text LLM — a direct, non-streaming Claude call that returns a string.

    LightRAG's bundled Anthropic wrapper always streams and returns an async
    generator, but entity extraction (and the default query path) expect a
    plain string, so the SDK is called directly here. Returns a coroutine for
    LightRAG to await. Unrecognised kwargs (LightRAG internals like hashing_kv)
    are ignored rather than forwarded to the API.
    """
    max_tokens = kwargs.pop("max_tokens", MAX_TOKENS)
    messages = list(history_messages or []) + [{"role": "user", "content": prompt}]

    async def _run():
        client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        params = {
            "model": LLM_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            params["system"] = system_prompt
        # Stream and collect into a string: the SDK refuses a non-streaming
        # request whose max_tokens could take over 10 minutes, and entity
        # extraction uses a large max_tokens. Streaming returns the same text.
        parts = []
        async with client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                parts.append(text)
        return "".join(parts)

    return _run()


async def _claude_vision(prompt, system_prompt=None, image_data=None, **kwargs):
    """Call Claude directly with an image content block."""
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    mime_type = (
        image_mime_type(base64.b64decode(image_data[:24])) or "image/jpeg"
        if image_data
        else None
    )
    content = []
    if image_data:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_data,
                },
            }
        )
    content.append({"type": "text", "text": prompt})

    params = {
        "model": VISION_MODEL,
        "max_tokens": kwargs.pop("max_tokens", 1024),
        "messages": [{"role": "user", "content": content}],
    }
    if system_prompt:
        params["system"] = system_prompt

    response = await client.messages.create(**params)
    return response.content[0].text


def vision_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    """
    Vision LLM for RAGAnything.

    - 'messages' format (VLM enhanced query): rebuild for Anthropic's content-block API.
    - 'image_data' (single image from parser): call Claude with a native image block.
    - Text-only fallback: delegate to llm_model_func.
    """
    if messages:
        # RAGAnything passes OpenAI-style messages; convert to Anthropic format.
        anthropic_messages = []
        for m in messages:
            if m is None:
                continue
            role = m.get("role", "user")
            if role == "system":
                # Anthropic uses a top-level 'system' param — skip here, handled below
                continue
            raw = m.get("content", "")
            if isinstance(raw, str):
                anthropic_messages.append({"role": role, "content": raw})
            elif isinstance(raw, list):
                # Convert OpenAI image_url blocks to Anthropic image blocks
                blocks = []
                for block in raw:
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "image_url":
                        url = block["image_url"]["url"]
                        if url.startswith("data:"):
                            # data:<mime>;base64,<data>
                            header, b64 = url.split(",", 1)
                            mime = header.split(":")[1].split(";")[0]
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime,
                                        "data": b64,
                                    },
                                }
                            )
                anthropic_messages.append({"role": role, "content": blocks})

        # Extract system from original messages
        sys_text = next(
            (m["content"] for m in messages if m and m.get("role") == "system"), None
        )

        async def _run():
            client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            params = {
                "model": VISION_MODEL,
                "max_tokens": kwargs.pop("max_tokens", 1024),
                "messages": anthropic_messages,
            }
            if sys_text:
                params["system"] = sys_text
            resp = await client.messages.create(**params)
            return resp.content[0].text

        return _run()

    elif image_data:
        return _claude_vision(
            prompt, system_prompt=system_prompt, image_data=image_data, **kwargs
        )
    else:
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)


embedding_func = EmbeddingFunc(
    embedding_dim=1024,  # bge-m3 default
    max_token_size=8192,
    func=lambda texts, **kw: ollama_embed(
        texts,
        embed_model=OLLAMA_EMBED_MODEL,
        host=OLLAMA_HOST,
    ),
)


async def run(
    file_path: str,
    output_dir: str = "./output",
    working_dir: str = "./rag_storage",
    queries: list[str] | None = None,
):
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=os.getenv("PARSER", _DEFAULT_PARSER),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Processing: {file_path}")
    await rag.process_document_complete(file_path=file_path, output_dir=output_dir)

    queries = queries or [
        "What is the main content of the document?",
        "What are the key topics discussed?",
    ]
    for q in queries:
        logger.info(f"\n[Query] {q}")
        result = await rag.aquery(q, mode="hybrid")
        logger.info(f"Answer: {result}")


if __name__ == "__main__":
    import argparse

    configure_logging()

    parser = argparse.ArgumentParser(
        description="RAGAnything — Claude LLM with Ollama embeddings"
    )
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--output", "-o", default=os.getenv("OUTPUT_DIR", "./output"))
    parser.add_argument("--working-dir", "-w", default="./rag_storage")
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        help="Ask a specific question (repeatable). "
        "Defaults to two generic queries when omitted.",
    )
    args = parser.parse_args()

    print("RAGAnything — Claude LLM with Ollama embeddings")
    print("=" * 40)
    asyncio.run(run(args.file_path, args.output, args.working_dir, args.query))
