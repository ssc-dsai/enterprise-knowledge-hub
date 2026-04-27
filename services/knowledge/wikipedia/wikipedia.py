"""Wikipedia knowledge service implementation.
    has custom way of ingesting data (from wikimedia dumps in bz2 format).
    has vectorization processing logic at the process step. to a vector db
"""
import bz2
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import hashlib

import numpy as np
from dotenv import load_dotenv
from wikitextparser import remove_markup

from provider.embedding.qwen3.embedder_factory import get_embedder
from repository.model import WikipediaDbRecord
from services.database.knowledge_item_service import KnowledgeItemService
from services.knowledge.base import KnowledgeService
from services.knowledge.batch_handler import BatchHandler
from services.knowledge.models import KnowledgeItem
from services.knowledge.wikipedia.models import WikipediaItemProcessed, Source, WikipediaItemRaw

load_dotenv()

PROGRESS_SUFFIX: str = ".progress"
INDEX_FILENAME = re.compile(r"(?P<prefix>.+)-index(?P<chunk>\d*)\.txt\.bz2")
QUEUE_BATCH_NAME = "wikipedia_embeddings_sink"

# Regex to detect internal wikilinks.
# Per https://www.mediawiki.org/wiki/Manual:Article_count a page is an "article" only if it
# contains at least one internal wikilink (https://www.mediawiki.org/wiki/Help:Links#Internal_links).
_ARTICLE_WIKILINK_RE = re.compile(r'\[\[[^\]]+\]\]')

@dataclass
class WikipediaKnowledgeService(KnowledgeService):
    """Knowledge service for Wikipedia ingestion."""

    _content_folder_path: Path = Path(os.getenv("WIKIPEDIA_CONTENT_FOLDER",
                                    "./content/wikipedia")).expanduser().resolve()
    _progress_flush_interval: int = 1000 # for the .progress file we track the line number we stopped at
    _batch_size: int = int(os.getenv("WIKIPEDIA_PROCESS_BATCH_SIZE", "256"))
    _debug_extraction: bool = os.getenv("DEBUG_EXTRACTION", "false").lower() in ("1", "true", "yes")

    def __init__(self, queue_service, logger, run_history_service):
        super().__init__(queue_service=queue_service, logger=logger,
                         run_history_service=run_history_service, service_name="wikipedia")
        self._knowledge_wikipedia_service = KnowledgeItemService(logger)
        self._batch_handler_instance = None

    @property
    def embedder(self):
        """Get embedder"""
        return get_embedder()

    def get_batch_size(self):
        return self._batch_size

    def process_handler(self, item: KnowledgeItem, delivery_tag: str) -> bool:
        """
        Override base implementaiton
        Handler definition for process step — delegates to BatchHandler for batching.
        """
        if not hasattr(self, "_batch_handler_instance"):
            def acknowledge(dt: int, successful: bool):
                self.queue_service.read_ack(dt, successful)
            self._batch_handler_instance = BatchHandler(
                self.process_item, acknowledge, self.get_batch_size(), self.logger
            )
        self._batch_handler_instance(item, delivery_tag)

        # BatchHandler manages ack internally. Tells QueueWorker not to ack
        return False

    def finalize_process(self) -> None:
        """
        Optional hook from base.py
        Flush any remaining items in the batch before the process loop ends.
        """
        if hasattr(self, "_batch_handler_instance") and self._batch_handler_instance.item_list:
            self._batch_handler_instance.flush()


    def process_item(self, knowledge_item: list[KnowledgeItem]) -> list[WikipediaItemProcessed]:
        """Process ingested WikipediaItem from the queue and return one row per text chunk."""
        try:
            start_time = time.perf_counter()
            gpu_batch_size = self.embedder.get_batch_size()

            batch: list[str] = []
            for item in knowledge_item:
                batch.append(item['content'])

            # PLACEHOLDER for actual embedding generation. For now, we just generate dummy embeddings.
            # embeddings = [np.random.rand(512).tolist() for _ in batch]
            embeddings = self.embedder.embed(batch)
            arr = np.asarray(embeddings)

            results: list[WikipediaItemProcessed] = []

            for (item, vec) in zip(knowledge_item, arr):
                results.append(
                    WikipediaItemProcessed(
                        name=item['name'],
                        content=item['content'],
                        last_modified_date=item['last_modified_date'],
                        pid=item['pid'],
                        chunk_index=item['chunk_index'],
                        chunk_count=item['chunk_count'],
                        source=item['source'],
                        embeddings=vec,
                    )
                )
            for processed_item in results:
                self.emit_processed_item(processed_item)

            end_time = time.perf_counter()
            self.logger.info("Generated embeddings for %s items in %.2f seconds per batch (GPU batch size: %s)",
                             len(knowledge_item), (end_time - start_time)/gpu_batch_size, gpu_batch_size)

        except Exception as e:
            self.logger.error("Error processing embedding for Wikipedia item: %s", e)
            # logger debug for what item?
            raise e

    def fetch_from_source(self) -> Iterator[WikipediaItemRaw]:
        """Read data from Wikipedia index.txt.bz2 source.

            The content will be first entrypoint in the main .bz2 multistream file.
                Ex: 345:6789:Fruits (Read more on the doc from the README.md from content/ folder)
        """
        for index_path in self._discover_index_files():
            self.logger.info("Reading data from Wikipedia source: %s", index_path)

            dump_path = self._get_dump_path(index_path)
            if dump_path is None:
                continue

            try:
                yield from self._process_index_file(index_path, dump_path)
                # DONE INDEXING FILE X, call history service to report (future)
            except OSError as exc:
                self.logger.error("Failed to process index file %s: %s. Continuing to next file.", index_path, exc)
                continue

    def emit_fetched_item(self, item) -> None:
        max_tokens = getattr(self.embedder, "max_seq_length", None)

        # taking item and chunking them
        chunks = self.embedder.chunk_text_by_tokens(item.content, max_tokens=max_tokens)
        results: list[WikipediaItemRaw] = []
        num_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks, start=1):
            results.append(
                WikipediaItemRaw(
                    name=item.name,
                    content=chunk_text,
                    last_modified_date=item.last_modified_date,
                    pid=item.pid,
                    chunk_index=idx,
                    chunk_count=num_chunks,
                    source=item.source
                )
            )
        for wiki_item in results:
            self.queue_service.write(self._ingest_queue_name(), wiki_item)

    def _get_dump_path(self, index_path: Path) -> Path | None:
        """Derive the dump file path from an index file path."""
        match = INDEX_FILENAME.match(index_path.name)
        if not match:
            return None

        prefix = match.group("prefix")
        chunk = match.group("chunk")
        dump_name = f"{prefix}{chunk if chunk else ''}.xml.bz2"
        dump_path = index_path.with_name(dump_name)

        if not dump_path.exists():
            self.logger.warning("Dump file not found: %s", dump_path)
            return None

        return dump_path

    def emit_processed_item(self, item: WikipediaItemProcessed) -> None:
        self.queue_service.write(self._processed_queue_name(), item)

    def store_item(self, item: WikipediaItemProcessed) -> None:
        item_validated = WikipediaItemProcessed.model_validate(item)
        record_to_insert = WikipediaDbRecord.from_item(item_validated)
        self._knowledge_wikipedia_service.insert(record_to_insert.as_mapping())

    def compute_run_id(self, files: list[Path]) -> int:
        """
        Gets a unique run_id based on files in folder and time
        32-bit hash truncated to fit PostgreSQL INTEGER (31-bit signed pos range)
        Hash based on filename, size and timestamp (.name, .st_size, .st_mtime_ns)
        """
        run_id_hash = hashlib.sha256()

        for file in sorted(files):
            stat = file.stat()
            run_id_hash.update(file.name.encode())
            run_id_hash.update(str(stat.st_size).encode())
            run_id_hash.update(str(stat.st_mtime_ns).encode())

        # I dont want to change column in pg, so truncating to 32 bits (4 bytes).
        # I know.  small chance of collision.  But we're not running in the millions (assuming 1 run a month)
        # & 0x7FFFFFFF is to truncate to fit into PG integer (31 bit)
        return int.from_bytes(run_id_hash.digest()[:4], "big", signed=False) & 0x7FFFFFFF

    def _get_run_id(self) -> int:
        """
        Get unique run_id based on index files provided in content folder
        """
        index_files: list[Path] = []
        for index_path in self._discover_index_files():
            index_files.append(index_path)

        return self.compute_run_id(index_files)

    def _process_index_file(self, index_path: Path, dump_path: Path) -> Iterator[WikipediaItemRaw]: #pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Process a single index file and yield WikipediaItems."""
        start_line = self._load_progress(index_path)
        if start_line > 0:
            self.logger.info("Resuming from line %d for %s", start_line, index_path.name)

        current_line = 0

        source_name = index_path.name .split("-")[0]
        source = Source.WIKIPEDIA_EN if source_name == "enwiki" else Source.WIKIPEDIA_FR if source_name == "frwiki" else None #pylint: disable=line-too-long

        with open(dump_path, 'rb') as dump_file, bz2.open(index_path, mode='rt') as index_file:
            line_iter = iter(index_file)
            line = next(line_iter, None)
            last_offset = None
            while line is not None:
                next_line = next(line_iter, None)

                current_line += 1
                line_offset = self._parse_line_offset(line, current_line, index_path.name)
                if line_offset is None: # if we cannot read the current line skip to the next already..
                    line = next_line
                    continue

                # adding a clause that sets the last_offset for the first time.
                if last_offset is None:
                    last_offset = line_offset

                # Skip already processed lines (update byte offset as we go)
                if current_line <= start_line:
                    last_offset = line_offset
                    line = next_line
                    continue

                # if the current byte offset is different than the last byte offset it means we need to extract from bz2
                if last_offset != line_offset:
                    yield from self._process_chunk(dump_file, dump_path.name, last_offset, line_offset, source)
                    last_offset = line_offset

                # Last item on the list clause... need to extract until the end of the bz2 archive..
                if next_line is None:
                    yield from self._process_chunk(dump_file, dump_path.name, line_offset, None, source)

                if current_line % self._progress_flush_interval == 0:
                    self._save_progress(index_path, current_line)

                line = next_line

        self._save_progress(index_path, current_line)
        self.logger.info("Completed %s at line %d", index_path.name, current_line)

    def _parse_line_offset(self, line: str, line_num: int, filename: str) -> int | None:
        """Parse the byte offset from an index line. Returns None if malformed."""
        try:
            offset_str, _, _ = line.strip().split(":", 2)
            return int(offset_str)
        except ValueError:
            self.logger.warning("Skipping malformed line %d in %s", line_num, filename)
            return None

    def _process_chunk( #pylint: disable=too-many-arguments,too-many-positional-arguments
        self, dump_file, dump_name: str, prev_offset: int , offset: int | None, source: Source | None
    ) -> Iterator[WikipediaItemRaw]:
        """Decompress and parse a chunk of the dump file."""
        dump_file.seek(prev_offset) # read from where we left off.
        # If we have an offset if means we need to stop somewhere.
        if offset is not None:
            length = offset - prev_offset
            data = dump_file.read(length)
        else:
            data = dump_file.read() # read until the end of the file.

        try:
            decompressed = bz2.decompress(data)
            xml_content = decompressed.decode("utf-8", errors="ignore")
            yield from self._extract_pages_from_xml(xml_content, source)
        except Exception as exc:
            self.logger.error(
                "Failed to decompress chunk from %s between offsets %s and %s: %s",
                dump_name, prev_offset, offset, exc
            )

    def _extract_pages_from_xml(self, xml_content: str, source: Source | None) -> Iterator[WikipediaItemRaw]:
        """Extract and parse all pages from XML content."""
        for page_match in re.finditer(r"<page>(.*?)</page>", xml_content, re.DOTALL):
            page_xml = page_match.group(0)

            # optional write to disk for debug purposes
            if self._debug_extraction:
                try:
                    title_match = re.search(r"<title>([^<]+)</title>", page_xml)
                    title = title_match.group(1) if title_match else "unknown_title"
                    safe_title = re.sub(r'[\\/:"*?<>|]+', '_', title)  # Sanitize filename
                    debug_path = self._content_folder_path / "debug_extracted_pages"
                    debug_path.mkdir(parents=True, exist_ok=True)
                    file_path = debug_path / f"{safe_title}.xml"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(page_xml)
                except Exception as exc:
                    self.logger.debug("Failed to write extracted page xml: %s", exc)

            item = self._parse_page_xml(page_xml)
            if item:
                item.source = source
                if not self._should_ignore_page(item):

                    # REMOVE WIKI MARKUP (note: one of those 2 methods might be faster than the other??
                    # they yield the same results)
                    item.content = remove_markup(item.content)
                    # content = parse(content).plain_text()

                    # if item is to be processed, we need to ensure we delete
                    # any existing record of "older" version of this article
                    self._knowledge_wikipedia_service.delete_by_pid_source(item.pid, item.source)
                    yield item

    def _save_progress(self, index_path: Path, line_number: int) -> None:
        """Save current progress (line number) to a small file."""
        progress_path = index_path.with_suffix(index_path.suffix + PROGRESS_SUFFIX)
        progress_path.write_text(str(line_number))

    def _load_progress(self, index_path: Path) -> int:
        """Load the last processed line number. Returns 0 if no progress file exists."""
        progress_path = index_path.with_suffix(index_path.suffix + PROGRESS_SUFFIX)
        if not progress_path.exists():
            return 0
        try:
            return int(progress_path.read_text().strip())
        except (ValueError, OSError):
            return 0

    def _discover_index_files(self) -> Iterator[Path]:
        """Discover index files in the content folder."""
        self.logger.debug("Searching for index files in ---> %s", self._content_folder_path)
        for node in sorted(self._content_folder_path.rglob("*.txt.bz2")):
            if not node.is_file():
                continue
            if node.suffix == PROGRESS_SUFFIX:
                continue  # Skip progress files
            # Use fullmatch to ensure entire filename matches (excludes :Zone.Identifier files)
            match = INDEX_FILENAME.fullmatch(node.name)
            if not match:
                self.logger.debug("Skipping index file with unknown pattern: %s", node.name)
                continue
            yield node

    def _should_ignore_page(self, page: WikipediaItemRaw) -> bool:
        """Check if a page should be ignored based on title or type."""

        #Namespace detection: https://en.wikipedia.org/wiki/Wikipedia:Namespace
        if not page.is_namespace_0:
            return True

        if page.is_redirect:
            return True

        if not page.has_wikilinks:
            return True

        # DB metadata check, last resort before we do in fact process the item.
        if self._knowledge_wikipedia_service.record_is_up_to_date(page.pid, page.source, page.last_modified_date):
            return True

        return False

    def _parse_page_xml(self, xml_page: str) -> WikipediaItemRaw | None:
        """Parse a Wikipedia page XML and extract relevant fields."""

        match = re.search(
            r"<title>(?P<title>[^<]+)</title>.*?"
            r"<id>(?P<pid>\d+)</id>.*?"
            r"<text[^>]*>(?P<text>[^<]*(?:<(?!/text>)[^<]*)*)</text>",
            xml_page,
            re.DOTALL,
        )

        if not match:
            return None

        title = match.group("title") # Extract title
        pid= match.group("pid") # Extract page ID
        content = match.group("text") # Extract content (wiki markup text)

        # Detect internal wikilinks BEFORE stripping markup (Wikipedia requires >=1 to count as "article")
        # Excludes Category/File/Image links which don't count toward the pagelinks table
        has_wikilinks = bool(_ARTICLE_WIKILINK_RE.search(content))

        # Detect content-based redirects (#REDIRECT in wikitext) before markup removal destroys the marker
        is_content_redirect = content.lstrip().upper().startswith("#REDIRECT")

        # Extract last modified date (timestamp)
        timestamp_match = re.search(r"<timestamp>([^<]+)</timestamp>", xml_page)
        last_modified_date = None
        if timestamp_match:
            try:
                last_modified_date = datetime.fromisoformat(timestamp_match.group(1).replace("Z", "+00:00"))
            except ValueError:
                pass

        if not title or not content:
            return None

        return WikipediaItemRaw(
            name=title,
            content=content,
            last_modified_date=last_modified_date,
            pid=pid,
            is_namespace_0=bool(re.search(r"<ns>0</ns>", xml_page)),
            is_redirect=(bool(re.search(r"<redirect\s", xml_page)) or is_content_redirect),
            has_wikilinks=has_wikilinks,
        )
