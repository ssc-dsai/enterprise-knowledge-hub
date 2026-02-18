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

import numpy as np
from dotenv import load_dotenv
from wikitextparser import remove_markup

from provider.embedding.qwen3.embedder_factory import get_embedder
from repository.postgrespg import WikipediaDbRecord, WikipediaPgRepository
from services.knowledge.base import KnowledgeService
from services.knowledge.models import DatabaseWikipediaItem, Source, WikipediaItem

load_dotenv()

PROGRESS_SUFFIX: str = ".progress"
INDEX_FILENAME = re.compile(r"(?P<prefix>.+)-index(?P<chunk>\d*)\.txt\.bz2")
QUEUE_BATCH_NAME = "wikipedia_embeddings_sink"

@dataclass
class WikipediaKnowedgeService(KnowledgeService):
    """Knowledge service for Wikipedia ingestion."""

    _ignored_title_prefixes: tuple[str, ...] = (
            "Draft:",
            "Category:",
            "File:",
            "Wikipedia:",
            "Ébauche:",
            "Catégorie:",
            "Fichier:",
            "Wikipédia:",
            "Portal:",
            "Portail:",
            "Template:",
            "Modèle:",
            "Help:",
            "Aide:",
            "User:",
            "Utilisateur:",
            "Project:",
            "Projet:",
        )

    _content_folder_path: Path = Path(os.getenv("WIKIPEDIA_CONTENT_FOLDER",
                                    "./content/wikipedia")).expanduser().resolve()
    _process_only_first_n_paragraphs: int = int(os.getenv("WIKIPEDIA_PROCESS_ONLY_FIRST_N_PARAGRAPHS", "0"))
    _progress_flush_interval: int = 1000 # for the .progress file we track line number we stpped.
    _batch_size: int = int(os.getenv("POSTGRES_BATCH_SIZE", "500"))
    _debug_extraction: bool = os.getenv("DEBUG_EXTRACTION", "false").lower() in ("1", "true", "yes")

    def __init__(self, queue_service, logger, repository: WikipediaPgRepository | None = None):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")
        self._repository = repository or WikipediaPgRepository.from_env()
        self._pending: list[WikipediaDbRecord] = []

    @property
    def embedder(self):
        """Get embedder"""
        return get_embedder()

    def process_item(self, knowledge_item: dict[str, object]) -> list[DatabaseWikipediaItem]:
        """Process ingested WikipediaItem from the queue and return one row per text chunk."""
        try:
            # print(f"Processing item: {knowledge_item.get('title', 'unknown_title')}")
            item = WikipediaItem.from_dict(knowledge_item)
            self.logger.debug("Generating embeddings for %s", item.title)

            max_tokens = getattr(self.embedder, "max_seq_length", None)
            if max_tokens is None:
                max_tokens = getattr(getattr(self.embedder, "model", None), "max_seq_length", None)

            chunks = self.embedder.chunk_text_by_tokens(item.content, max_tokens=max_tokens)
            # PLACEHOLDER for actual embedding generation, which should be done in batches for efficiency. For now, we just generate dummy embeddings.
            embeddings = [np.random.rand(512).tolist() for _ in chunks]
            # embeddings = self.embedder.embed(item.content)

            arr = np.asarray(embeddings)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            num_chunks = arr.shape[0]
            if num_chunks != len(chunks):
                self.logger.warning("Chunk/text count mismatch: embeddings=%d, chunks=%d", num_chunks, len(chunks))
                limit = min(num_chunks, len(chunks))
                arr = arr[:limit]
                chunks = chunks[:limit]
                num_chunks = limit

            results: list[DatabaseWikipediaItem] = []
            for idx, (chunk_text, vec) in enumerate(zip(chunks, arr), start=1):
                results.append(
                    DatabaseWikipediaItem(
                        name=item.name,
                        title=f"{item.title} (chunk {idx}/{num_chunks})",
                        content=chunk_text,
                        last_modified_date=item.last_modified_date,
                        pid=item.pid,
                        chunk_index=idx,
                        chunk_count=num_chunks,
                        source=item.source,
                        embeddings=vec,
                    )
                )

            return results
        except Exception as e:
            self.logger.error("Error processing embedding for Wikipedia item: %s", e)
            raise e

    def fetch_from_source(self) -> Iterator[WikipediaItem]:
        """Read data from Wikipedia index.txt.bz2 source.

            The content will be first entrypoint in the main .bz2 multistream file.
                Ex: 345:6789:Fruits (Read more on the doc from the README.md from content/ folder)
        """
        print(f"Looking for index files in {self._content_folder_path}")
        for index_path in self._discover_index_files():
            print(f"Found index file: {index_path}")
            self.logger.info("Reading data from Wikipedia source: %s", index_path)

            dump_path = self._get_dump_path(index_path)
            if dump_path is None:
                continue

            try:
                yield from self._process_index_file(index_path, dump_path)
            except OSError as exc:
                self.logger.error("Failed to process index file %s: %s. Continuing to next file.", index_path, exc)
                continue

    def emit_fetched_item(self, item) -> None:
        self.queue_service.write(self._ingest_queue_name(), item.to_dict())

    def _get_dump_path(self, index_path: Path) -> Path | None:
        """Derive the dump file path from an index file path."""
        match = INDEX_FILENAME.match(index_path.name)
        if not match:
            return None

        prefix = match.group("prefix")
        chunk = match.group("chunk")
        dump_name = f"{prefix}{chunk if chunk else ''}.xml.bz2"
        dump_path = index_path.with_name(dump_name)

        # print(f"dump_name: {dump_name}")
        # print(f"dump_path: {dump_path}")
        # now = datetime.now()
        # self._repository.update_history_table_start(now, dump_name)

        if not dump_path.exists():
            self.logger.warning("Dump file not found: %s", dump_path)
            return None

        return dump_path

    def emit_processed_item(self, item: DatabaseWikipediaItem) -> None:
        queue_item = item.to_dict()
        self.queue_service.write(self._processed_queue_name(), queue_item)

    def store_item(self, item: dict[str, object]) -> None:
        print(f"Storing item: {item.get('title', 'unknown_title')}")
        wiki_item = DatabaseWikipediaItem.from_rabbitqueue_dict(item)
        record_to_insert = WikipediaDbRecord.from_item(wiki_item)
        self._repository.insert(record_to_insert.as_mapping())
        # print(f"Finished storing item: {item.get('title', 'unknown_title')}")

    def _process_index_file(self, index_path: Path, dump_path: Path) -> Iterator[WikipediaItem]: #pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Process a single index file and yield WikipediaItems."""
        start_line = self._load_progress(index_path)
        if start_line > 0:
            self.logger.info("Resuming from line %d for %s", start_line, index_path.name)

        current_line = 0

        source_name = index_path.name .split("-")[0]
        source = Source.WIKIPEDIA_EN if source_name == "enwiki" else Source.WIKIPEDIA_FR if source_name == "frwiki" else None #pylint: disable=line-too-long

        print(f"Processing index file: {index_path} with dump file: {dump_path} (source: {source})")
        now = datetime.now()
        history_id = self._repository.update_history_table_start(now, dump_path.name)

        with open(dump_path, 'rb') as dump_file, bz2.open(index_path, mode='rt') as index_file:
            line_iter = iter(index_file)
            line = next(line_iter, None)
            last_offset = 0
            while line is not None:
                next_line = next(line_iter, None)
                is_last = next_line is None

                current_line += 1
                line_offset = self._parse_line_offset(line, current_line, index_path.name)
                if line_offset is None:
                    line = next_line
                    continue

                # Skip already processed lines
                if current_line <= start_line:
                    last_offset = line_offset
                    line = next_line
                    continue

                if last_offset != line_offset or is_last:
                    yield from self._process_chunk(dump_file, dump_path.name, last_offset, line_offset, source)

                if current_line % self._progress_flush_interval == 0:
                    self._save_progress(index_path, current_line)

                last_offset = line_offset
                line = next_line

        self._save_progress(index_path, current_line)
        self.logger.info("Completed %s at line %d", index_path.name, current_line)

        self._repository.update_history_table_end(datetime.now(), history_id)

    def _parse_line_offset(self, line: str, line_num: int, filename: str) -> int | None:
        """Parse the byte offset from an index line. Returns None if malformed."""
        try:
            offset_str, _, _ = line.strip().split(":", 2)
            return int(offset_str)
        except ValueError:
            self.logger.warning("Skipping malformed line %d in %s", line_num, filename)
            return None

    def _process_chunk( #pylint: disable=too-many-arguments,too-many-positional-arguments
        self, dump_file, dump_name: str, prev_offset: int | None, offset: int, source: Source | None
    ) -> Iterator[WikipediaItem]:
        """Decompress and parse a chunk of the dump file."""

        length = offset - prev_offset
        dump_file.seek(prev_offset)
        data = dump_file.read(length)

        try:
            decompressed = bz2.decompress(data)
            xml_content = decompressed.decode("utf-8", errors="ignore")
            yield from self._extract_pages_from_xml(xml_content, source)
        except OSError as exc:
            self.logger.error(
                "Failed to decompress chunk from %s between offsets %s and %s: %s",
                dump_name, prev_offset, offset, exc
            )

    def _extract_pages_from_xml(self, xml_content: str, source: Source | None) -> Iterator[WikipediaItem]:
        """Extract and parse all pages from XML content."""
        print(f"Extracting pages from XML content (source: {source})")
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

            if not self._should_ignore_page(page_xml):
                item = self._parse_page_xml(page_xml)
                if item:
                    item.source = source
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
        print(f"Searching for index files in ---> {self._content_folder_path}")
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

    def _should_ignore_page(self, xml_page: str) -> bool:
        """Check if a page should be ignored based on title or type."""

        #Namespace detection: https://en.wikipedia.org/wiki/Wikipedia:Namespace
        if not re.search(r"<ns>0</ns>", xml_page):
            return True

        if re.search(r"<redirect\s", xml_page):
            return True

        # last resort, extra title checking
        title_match = re.search(r"<title>([^<]+)</title>", xml_page)
        if title_match:
            title = title_match.group(1)
            for prefix in self._ignored_title_prefixes:
                if title.startswith(prefix):
                    return True
        return False

    def _parse_page_xml(self, xml_page: str) -> WikipediaItem | None:
        """Parse a Wikipedia page XML and extract relevant fields."""
        # Extract title
        title_match = re.search(r"<title>([^<]+)</title>", xml_page)
        title = title_match.group(1) if title_match else ""

        # Extract page ID
        pid_match = re.search(r"<id>(\d+)</id>", xml_page)
        pid = int(pid_match.group(1)) if pid_match else 0

        # Extract content (wiki markup text)
        text_match = re.search(r"<text[^>]*>([^<]*(?:<(?!/text>)[^<]*)*)</text>", xml_page, re.DOTALL)
        content = text_match.group(1) if text_match else ""

        # REMOVE WIKI MARKUP (note: one of those 2 methods might be faster than the other?? they yield the same results)
        content = remove_markup(content)
        #content = parse(content).plain_text()

        if self._process_only_first_n_paragraphs > 0:
            # untested bit of code ... to be tweaked, online it says a line is needed for markdown to do a
            # paragraph break, so just using \n for this ...
            paragraphs = re.split(r'\n{2,}', content)
            content = '\n\n'.join(paragraphs[:self._process_only_first_n_paragraphs])

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

        return WikipediaItem(
            name=title,
            title=title,
            content=content,
            last_modified_date=last_modified_date,
            pid=pid,
        )
