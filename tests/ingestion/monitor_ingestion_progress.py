"""Start an ingestion run and stream stage progress metadata to CSV.

Usage example:
    uv run --env-file .env python tests/ingestion/monitor_ingestion_progress.py \
        --service tbs-policies \
        --api-base-url http://localhost:8000 \
        --poll-interval 2.0 \
        --timeout-seconds 7200

    This will start a new run for the tbs-policies service,
    and poll the run_history table every 2 seconds for up to a limit of 2 hours
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import psycopg
from dotenv import load_dotenv

# Constants for run_history status values used to detect run completion
RUN_ENDED = "Run Completed"
RUN_STOPPED = "Run Manually Stopped"


@dataclass
class ProgressRow:
    """Projected run_history row for CSV output."""
    status: str
    metadata: dict[str, Any] | None


def _parse_metadata(raw: Any) -> dict[str, Any] | None:
    """Parse run_history metadata."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        try:
            value = ast.literal_eval(text)
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            pass
    return None


def _open_db_connection() -> psycopg.Connection:
    """Open a PostgreSQL connection using the .env file variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "rag")
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "admin")

    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
    )


def _fetch_run_rows(conn: psycopg.Connection, run_id: int, service_name: str) -> list[ProgressRow]:
    """Fetch all run_history rows for a run."""
    sql = (
        "SELECT status, metadata "
        "FROM run_history "
        "WHERE run_id = %s AND service_name = %s "
        "ORDER BY timestamp ASC"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (run_id, service_name))
        rows = cur.fetchall()

    result: list[ProgressRow] = []
    for row in rows:
        result.append(
            ProgressRow(
                status=row[0],
                metadata=_parse_metadata(row[1]),
            )
        )
    return result


def _fetch_stage_metrics(conn: psycopg.Connection, run_id: int, service_name: str) -> list[dict[str, Any]]:
    """Fetch stage progress metadata from run_metrics (where completed/total/throughput actually live)."""
    sql = (
        "SELECT rm.metadata "
        "FROM run_metrics rm "
        "JOIN run_history rh ON rh.id = rm.run_history_id "
        "WHERE rh.run_id = %s AND rh.service_name = %s "
        "ORDER BY rm.timestamp ASC"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (run_id, service_name))
        rows = cur.fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        metadata = _parse_metadata(row[0])
        if metadata is not None:
            result.append(metadata)
    return result


def _start_run(api_base_url: str, service: str, run_id: int) -> None:
    """Call the run endpoint with a fixed run_id."""
    url = f"{api_base_url.rstrip('/')}/knowledge/{service}/run?run_id={run_id}"
    with urlopen(url, timeout=30) as response:  # nosec B310
        _ = response.read()


def _stop_run(api_base_url: str, service: str) -> None:
    """Call the stop endpoint to halt the current run."""
    url = f"{api_base_url.rstrip('/')}/knowledge/{service}/stop"
    with urlopen(url, timeout=30) as response:  # nosec B310
        _ = response.read()


def _prepare_csv(path: Path) -> None:
    """Create CSV parent directory and write header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sample_time_utc",
                "experiment_name",
                "run_id",
                "service_name",
                "stage",
                "stage_status",
                "completed",
                "total",
                "throughput",
                "elapsed_seconds",
                "database_volume_bytes",
            ]
        )


def _append_csv_row(path: Path, experiment_name: str | None, run_id: int, service_name: str,
                    stage: str, stage_status: str | None, completed: Any, total: Any,
                    throughput: Any, elapsed_seconds: Any, database_volume_bytes: int | None) -> None:
    """Append one sample to CSV."""
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                experiment_name,
                run_id,
                service_name,
                stage,
                stage_status,
                completed,
                total,
                throughput,
                elapsed_seconds,
                database_volume_bytes,
            ]
        )


def _build_default_output_path() -> Path:
    """Build shared default output location for multi-run comparisons."""
    return Path("tests") / "ingestion" / "results" / "ingestion_progress.csv"


def _fetch_database_volume_bytes(conn: psycopg.Connection) -> int | None:
    """Return current database size in bytes for coarse storage growth tracking."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_database_size(current_database())")
        row = cur.fetchone()
    if row is None:
        return None
    value = row[0]
    if value is None:
        return None
    return int(value)


def _append_stage_progress_rows(
    stage_metrics: list[dict[str, Any]],
    output_path: Path,
    experiment_name: str | None,
    run_id: int,
    service_name: str,
    database_volume_bytes: int | None,
) -> None:
    """Append all stage progress rows for the current poll sample."""
    for metadata in stage_metrics:
        _append_csv_row(
            output_path,
            experiment_name,
            run_id,
            service_name,
            metadata.get("stage"),
            metadata.get("status"),
            metadata.get("completed"),
            metadata.get("total"),
            metadata.get("throughput"),
            metadata.get("elapsed_seconds"),
            database_volume_bytes,
        )


def _monitor_run_progress(
    run_id: int,
    service_name: str,
    output_path: Path,
    experiment_name: str | None,
    poll_interval: float,
    timeout_seconds: float,
    api_base_url: str,
) -> int:
    """Poll run history and append stage progress rows until end or timeout."""
    started = time.perf_counter()

    with _open_db_connection() as conn:
        while True:
            rows = _fetch_run_rows(conn, run_id, service_name)
            stage_metrics = _fetch_stage_metrics(conn, run_id, service_name)
            database_volume_bytes = _fetch_database_volume_bytes(conn)
            ended = any(row.status in (RUN_ENDED, RUN_STOPPED) for row in rows)
            _append_stage_progress_rows(
                stage_metrics=stage_metrics,
                output_path=output_path,
                experiment_name=experiment_name,
                run_id=run_id,
                service_name=service_name,
                database_volume_bytes=database_volume_bytes,
            )

            if ended:
                print(f"Run {run_id} ended. CSV written to {output_path}")
                return 0

            if time.perf_counter() - started > timeout_seconds:
                print(f"Timed out after {timeout_seconds} seconds. Sending stop signal...")
                try:
                    _stop_run(api_base_url, service_name)
                except URLError as exc:
                    print(f"Failed to send stop signal: {exc}")
                print(f"Partial CSV at {output_path}")
                return 2

            time.sleep(poll_interval)


def main() -> int:
    """Entrypoint."""
    parser = argparse.ArgumentParser(description="Monitor EKH ingestion run progress to CSV")
    parser.add_argument("--service", default="tbs-policies", help="Source service, e.g. wikipedia|tbs-policies")
    parser.add_argument("--api-base-url", default="http://localhost:8000", help="EKH API base URL")
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name for grouping/comparison")
    parser.add_argument("--run-id", type=int, default=None, help="Optional run_id to use")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0, help="Stop after this many seconds")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    load_dotenv()

    # Use the provided run_id or generate a (somewhat) random run_id.
    # This removes the need to handle searching for the correct run_id.
    run_id = args.run_id if args.run_id is not None else int(time.time()) & 0x7FFFFFFF

    output_path = Path(args.output_csv) if args.output_csv else _build_default_output_path()

    _prepare_csv(output_path)

    try:
        _start_run(args.api_base_url, args.service, run_id)
        print(f"Started run for '{args.service}' with run_id={run_id}")
    except URLError as exc:
        print(f"Failed to start run: {exc}")
        return 1

    return _monitor_run_progress(
        run_id=run_id,
        service_name=args.service,
        output_path=output_path,
        experiment_name=args.experiment_name,
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        api_base_url=args.api_base_url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
