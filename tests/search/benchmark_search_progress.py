"""Benchmark search latency for a dimension-tagged tbs-policies source, logging to CSV.

Hits the running EKH API's generic /{slug}/search endpoint so the dimension is
resolved server-side via kb_source_registry (register_tbs_policies_dim_source.py) -
no dimensions param needed in the request.

Usage:
    uv run --env-file .env python tests/search/monitor_search_progress.py \
        --slug tbs-policies-384 --dimensions 384 --experiment-name dim-384 --repeat 20
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import load_dotenv


def _load_queries(path: Path) -> list[str]:
    """Load the fixed test-query set."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_csv(path: Path) -> None:
    """Create CSV parent directory and write header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "sample_time_utc", "experiment_name", "slug", "dimensions",
            "query", "iteration", "latency_seconds", "num_results", "top_similarity",
        ])


def _append_csv_row(path: Path, experiment_name: str | None, slug: str, dimensions: int,
                    query: str, iteration: int, latency_seconds: float,
                    num_results: int, top_similarity: float | None) -> None:
    """Append one sample to CSV."""
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), experiment_name, slug, dimensions,
            query, iteration, latency_seconds, num_results, top_similarity,
        ])


def _search(api_base_url: str, slug: str, query: str, limit: int) -> dict[str, Any]:
    """Call the generic search endpoint for a given slug."""
    params = urlencode({"query": query, "limit": limit})
    url = f"{api_base_url.rstrip('/')}/database/{slug}/search?{params}"
    with urlopen(url, timeout=60) as response:  # nosec B310
        return json.loads(response.read())


def main() -> int:
    """Entrypoint."""
    parser = argparse.ArgumentParser(description="Benchmark EKH search latency to CSV")
    parser.add_argument("--slug", required=True, help="Search registry slug, e.g. tbs-policies-384")
    parser.add_argument("--dimensions", type=int, required=True, help="Dimensions the slug was registered with")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--queries-file", default=str(Path("tests") / "search" / "search_queries.json"))
    parser.add_argument("--repeat", type=int, default=10, help="Number of passes over the full query set")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    load_dotenv()

    queries = _load_queries(Path(args.queries_file))
    output_path = (
        Path(args.output_csv) if args.output_csv
        else Path("tests") / "search" / "results" / "search_progress.csv"
    )
    _prepare_csv(output_path)

    for iteration in range(args.repeat):
        for query in queries:
            started = time.perf_counter()
            result = _search(args.api_base_url, args.slug, query, args.limit)
            latency = time.perf_counter() - started

            results = result.get("results", [])
            top_similarity = results[0]["similarity"] if results else None

            _append_csv_row(
                output_path, args.experiment_name, args.slug, args.dimensions,
                query, iteration, latency, len(results), top_similarity,
            )
            print(f"[{args.slug}] iter={iteration} query={query!r} latency={latency:.3f}s results={len(results)}")

    print(f"Done. CSV written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
