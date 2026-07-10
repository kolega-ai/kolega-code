#!/usr/bin/env python3
"""Run the local web-fetch retrieval/extraction corpus and print JSON results."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from kolega_code.agent.tool_backend.web_fetch.extractors import extractor_names
from kolega_code.agent.tool_backend.web_fetch.pipeline import LocalWebContentPipeline, WebContentError

DEFAULT_URLS = (
    "https://example.com",
    "https://docs.python.org/3/library/asyncio.html",
    "https://github.com/python/cpython",
    "https://en.wikipedia.org/wiki/Web_scraping",
    "https://api.github.com/repos/python/cpython",
    "https://raw.githubusercontent.com/python/cpython/main/README.rst",
    "https://arxiv.org/pdf/1706.03762",
    "https://www.npmjs.com/package/react",
)
EXPECTED_DIAGNOSTICS = {
    "https://www.npmjs.com/package/react": ("HTTP 403", "JavaScript"),
}


async def run(urls: list[str], extractor: str) -> int:
    pipeline = LocalWebContentPipeline()
    rows: list[dict[str, object]] = []
    for url in urls:
        started = time.monotonic()
        try:
            result = await pipeline.load(url, extractor_preference=extractor)
            rows.append(
                {
                    "url": url,
                    "ok": True,
                    "final_url": result.final_url,
                    "method": result.method,
                    "content_type": result.content_type,
                    "bytes": result.byte_count,
                    "characters": len(result.content),
                    "warnings": list(result.warnings),
                    "extractor_attempts": [
                        {
                            "name": attempt.name,
                            "score": attempt.score,
                            "characters": attempt.content_length,
                            "usable": attempt.usable,
                            "error": attempt.error,
                        }
                        for attempt in result.attempts
                    ],
                    "retrieval_seconds": round(result.retrieval_seconds, 3),
                    "processing_seconds": round(result.processing_seconds, 3),
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
        except WebContentError as exc:
            rows.append(
                {
                    "url": url,
                    "ok": False,
                    "error": str(exc),
                    "seconds": round(time.monotonic() - started, 3),
                }
            )
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    def acceptable(row: dict[str, object]) -> bool:
        if row["ok"]:
            return True
        expected = EXPECTED_DIAGNOSTICS.get(str(row["url"]), ())
        return any(fragment in str(row.get("error")) for fragment in expected)

    return 0 if all(acceptable(row) for row in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", default=list(DEFAULT_URLS))
    parser.add_argument("--extractor", choices=extractor_names(), default="auto")
    args = parser.parse_args()
    return asyncio.run(run(args.urls, args.extractor))


if __name__ == "__main__":
    raise SystemExit(main())
