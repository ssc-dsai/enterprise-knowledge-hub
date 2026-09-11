"""
TBS Policies knowledge service implementation.

Ingests policy documents from the Treasury Board of Canada Secretariat (TBS)
policy hierarchy page. Fetches the hierarchy at:
    https://www.tbs-sct.canada.ca/pol/hierarch-eng.aspx

Extracts all policy page IDs from <li id="idXXXX"> elements, then fetches
each individual policy page and yields it as a KnowledgeItem for downstream
processing (embedding) and storage.
"""
import hashlib
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from repository.knowledge_tbs_policies_model import KnowledgeBaseTBSPolicies
from services.database.tbs_policy_item_service import TBSPolicyItemService
from services.knowledge.base import KnowledgeService
from services.knowledge.models import KnowledgeItem
from services.knowledge.tbs_policies.models import TBSPolicyItemRaw, TBSPolicyItemProcessed

# TBS hierarchy page listing all policies
TBS_HIERARCHY_URL = "https://www.tbs-sct.canada.ca/pol/hierarch-eng.aspx"
# Individual policy page URL template (HTML view)
TBS_POLICY_PAGE_URL = "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id={page_id}"

# Regex to extract numeric page IDs from <a id="idXXXX"> elements
_PAGE_ID_RE = re.compile(r"^id(\d+)$")

# Polite delay between fetches (seconds)
_FETCH_DELAY = float(os.getenv("TBS_FETCH_DELAY", "1.0"))
_REQUEST_TIMEOUT = int(os.getenv("TBS_REQUEST_TIMEOUT", "30"))


@dataclass
class TBSPoliciesKnowledgeService(KnowledgeService):
    """Knowledge service for TBS policies."""

    def __init__(self, queue_service, logger, run_history_service):
        super().__init__(queue_service=queue_service, logger=logger,
                         run_history_service=run_history_service, service_name="tbs-policies")
        self._session: requests.Session | None = None
        self._tbs_policy_service = TBSPolicyItemService(logger)

    def get_query_instruction(self) -> str:
        return (
            "Instruct: Given a query, retrieve relevant Government of Canada policy and directive "
            "passages that answer the query\n"
            "Query: "
        )

    @property
    def session(self) -> requests.Session:
        """Lazy-initialized requests session with common headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "EnterpriseKnowledgeHub/1.0 (GC Internal)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-CA,en;q=0.9",
            })
        return self._session

    def get_batch_size(self) -> int:
        return int(os.getenv("TBS_PROCESS_BATCH_SIZE", "32"))

    def _get_run_id(self) -> int:
        """Generate a run ID based on the current date (one run per day is expected)."""
        today = datetime.now().strftime("%Y-%m-%d")
        digest = hashlib.sha256(f"tbs-policies-{today}".encode()).digest()
        return int.from_bytes(digest[:4], "big", signed=False) & 0x7FFFFFFF

    # ─── INGEST STAGE ────────────────────────────────────────────────────────────

    def fetch_from_source(self) -> Iterator[TBSPolicyItemRaw]:
        """Fetch TBS policy hierarchy, extract page IDs, then yield each policy page."""
        self.logger.info("Fetching TBS policy hierarchy from %s", TBS_HIERARCHY_URL)

        page_ids = self._fetch_page_ids()
        self.logger.info("Discovered %d policy page IDs from hierarchy.", len(page_ids))

        for page_id in page_ids:
            try:
                item = self._fetch_policy_page(page_id)
                if item is None:
                    continue

                # Skip pages already stored with the same or newer last_modified_date
                if self._tbs_policy_service.record_is_up_to_date(
                    item.page_id, item.source, item.last_modified_date
                ):
                    self.logger.debug("Page id=%d (%s) is up to date, skipping.", page_id, item.name)
                    continue

                # Delete stale chunks before re-ingesting updated page
                self._tbs_policy_service.delete_by_page_id_source(item.page_id, item.source)
                yield item
            except Exception:
                self.logger.exception("Failed to fetch TBS policy page id=%d, skipping.", page_id)
                continue

            # Be polite to the server
            time.sleep(_FETCH_DELAY)

    def _fetch_page_ids(self) -> list[int]:
        """Fetch hierarchy page and extract all policy page IDs from <a id='idXXXX'> elements."""
        response = self.session.get(TBS_HIERARCHY_URL, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        page_ids: list[int] = []
        for a in soup.find_all("a", id=True):
            match = _PAGE_ID_RE.match(a.get("id", ""))
            if match:
                page_ids.append(int(match.group(1)))

        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for pid in page_ids:
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)

        return unique_ids

    def _fetch_policy_page(self, page_id: int) -> TBSPolicyItemRaw | None:
        """Fetch a single policy page by ID and parse its main content."""
        url = TBS_POLICY_PAGE_URL.format(page_id=page_id)
        self.logger.debug("Fetching TBS policy page: %s", url)

        response = self.session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Extract the policy title
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else f"Policy {page_id}"

        # TBS policy pages place the numbered policy body in <div id="ps-doc">.
        # Fall back to <main> (the outer page container), then <body> as a last resort.
        content_area = (
            soup.find("div", id="ps-doc")
            or soup.find("main")
            or soup.find("body")
        )

        if content_area is None:
            self.logger.warning("Empty page for id=%d, skipping.", page_id)
            return None

        # Extract text content, stripping navigation/scripts
        for tag in content_area.find_all(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        content = content_area.get_text(separator="\n", strip=True)

        if not content or len(content.strip()) < 50:
            self.logger.debug("Skipping page id=%d (%s) — content too short.", page_id, title)
            return None

        # Try to get last modified date from page metadata
        last_modified = self._extract_last_modified(soup)

        return TBSPolicyItemRaw(
            name=title,
            content=content,
            page_id=page_id,
            source="tbs-policies",
            last_modified_date=last_modified,
        )

    def _extract_last_modified(self, soup: BeautifulSoup) -> datetime | None:
        """Try to extract a last modified date from the page metadata."""
        # <meta name="dcterms.modified"> carries the policy's own effective date — check first.
        meta = soup.find("meta", attrs={"name": "dcterms.modified"})
        if meta and meta.get("content"):
            try:
                return datetime.strptime(meta["content"], "%Y-%m-%d")
            except ValueError:
                pass

        # Fallback: <dt>Date modified</dt> / <dd><time> pattern used on older canada.ca pages.
        date_modified_dl = soup.find("dt", string=re.compile(r"Date modified", re.IGNORECASE))
        if date_modified_dl:
            dd = date_modified_dl.find_next_sibling("dd")
            if dd:
                time_tag = dd.find("time")
                text = time_tag.get_text(strip=True) if time_tag else dd.get_text(strip=True)
                try:
                    return datetime.strptime(text, "%Y-%m-%d")
                except ValueError:
                    pass

        return None

    # ─── EMIT / PROCESS / STORE (queue plumbing) ─────────────────────────────────

    def emit_fetched_item(self, item: TBSPolicyItemRaw) -> None:
        """Chunk the raw item and write chunks to the ingest queue."""
        # Lazy import: avoids GPU initialisation during ingest-only runs
        from provider.embedding.qwen3.embedder_factory import get_embedder  # pylint: disable=import-outside-toplevel
        embedder = get_embedder()

        max_tokens = getattr(embedder, "max_seq_length", None)
        chunks = embedder.chunk_text_by_tokens(item.content, max_tokens=max_tokens)
        num_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks, start=1):
            chunk_item = TBSPolicyItemRaw(
                name=item.name,
                content=chunk_text,
                page_id=item.page_id,
                source=item.source,
                last_modified_date=item.last_modified_date,
                chunk_index=idx,
                chunk_count=num_chunks,
            )
            self.queue_service.write(self._ingest_queue_name(), chunk_item)

    def process_item(self, knowledge_item: KnowledgeItem) -> None:
        """Process a single item — compute embeddings and emit to processed queue."""
        from provider.embedding.qwen3.embedder_factory import get_embedder  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        embedder = get_embedder()
        content = knowledge_item['content'] if isinstance(knowledge_item, dict) else knowledge_item.content

        embeddings = embedder.embed([content])
        vec = np.asarray(embeddings)[0]

        item_data = knowledge_item if isinstance(knowledge_item, dict) else knowledge_item.model_dump()
        processed = TBSPolicyItemProcessed(
            name=item_data['name'],
            content=item_data['content'],
            page_id=item_data['page_id'],
            source=item_data['source'],
            last_modified_date=item_data.get('last_modified_date'),
            chunk_index=item_data.get('chunk_index', 1),
            chunk_count=item_data.get('chunk_count', 1),
            embeddings=vec,
            embedding_dims=int(vec.shape[0]),
        )
        self.emit_processed_item(processed)

    def emit_processed_item(self, item: TBSPolicyItemProcessed) -> None:
        """Write processed item to the processed queue."""
        self.queue_service.write(self._processed_queue_name(), item)

    def store_item(self, item: KnowledgeItem) -> None:
        """Store the processed item into the database."""
        validated = TBSPolicyItemProcessed.model_validate(item)
        record_to_insert = KnowledgeBaseTBSPolicies.from_item(validated)
        self._tbs_policy_service.insert(record_to_insert.as_mapping())
