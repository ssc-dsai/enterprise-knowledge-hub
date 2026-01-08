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
from services.knowledge.base import KnowledgeService
from services.knowledge.models import DatabaseWikipediaItem, WikipediaItem
from services.db.postgrespg import WikipediaDbRecord, WikipediaPgRepository

load_dotenv()

PROGRESS_SUFFIX: str = ".progress"
INDEX_FILENAME = re.compile(r"(?P<prefix>.+)-index(?P<chunk>\d*)\.txt\.bz2")

if os.getenv("WIKIPEDIA_EMBEDDING_MODEL_BACKEND", "LLAMA").upper() == "SENTENCE_TRANSFORMER":
    from provider.embedding.qwen3.sentence_transformer import Qwen3SentenceTransformer
    embedder = Qwen3SentenceTransformer()
else:
    from provider.embedding.qwen3.llama_embed import Qwen3LlamaCpp
    embedder = Qwen3LlamaCpp()
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
            "Portal:"
        )

    _content_folder_path: Path = Path(os.getenv("WIKIPEDIA_CONTENT_FOLDER",
                                    "./content/wikipedia")).expanduser().resolve()
    _process_only_first_n_paragraphs: int = int(os.getenv("WIKIPEDIA_PROCESS_ONLY_FIRST_N_PARAGRAPHS", "0"))
    _progress_flush_interval: int = 1000 # for the .progress file we track line number we stpped.
    _batch_size: int = int(os.getenv("POSTGRES_BATCH_SIZE", "500"))

    def __init__(self, queue_service, logger, repository: WikipediaPgRepository | None = None):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")
        self._repository = repository or WikipediaPgRepository.from_env()
        self._pending: list[WikipediaDbRecord] = []

    def process_queue(self, knowledge_item: dict[str, object]) -> list[DatabaseWikipediaItem]:
        """Process ingested WikipediaItem from the queue and return one row per text chunk."""
        try:
            item = WikipediaItem.from_dict(knowledge_item)
            self.logger.debug("Generating embeddings for %s", item.title)

            max_tokens = getattr(embedder, "max_seq_length", None)
            if max_tokens is None:
                max_tokens = getattr(getattr(embedder, "model", None), "max_seq_length", None)

            chunks = embedder.chunk_text_by_tokens(item.content, max_tokens=max_tokens)
            embeddings = embedder.embed(item.content)

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
        for index_path in self._discover_index_files():
            self.logger.info("Reading data from Wikipedia source: %s", index_path)

            dump_path = self._get_dump_path(index_path)
            if dump_path is None:
                continue

            try:
                yield from self._process_index_file(index_path, dump_path)
            except OSError as exc:
                self.logger.error("Failed to process index file %s: %s. Continuing to next file.", index_path, exc)
                continue

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

    def store_item(self, item: DatabaseWikipediaItem) -> None:
        """Store the processed knowledge item into the knowledge base."""
        record = WikipediaDbRecord.from_item(item)
        self._pending.append(record)
        if len(self._pending) >= self._batch_size:
            self._flush_pending()

    def finalize_processing(self) -> None:
        self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        self.logger.debug("Flushing %d wikipedia records to Postgres", len(self._pending))
        self._repository.insert_many(self._pending)
        self._pending.clear()

    def _process_index_file(self, index_path: Path, dump_path: Path) -> Iterator[WikipediaItem]:
        """Process a single index file and yield WikipediaItems."""
        start_line = self._load_progress(index_path)
        if start_line > 0:
            self.logger.info("Resuming from line %d for %s", start_line, index_path.name)

        current_line = 0
        prev_offset: int | None = None

        with open(dump_path, 'rb') as dump_file, bz2.open(index_path, mode='rt') as index_file:
            for line in index_file:
                current_line += 1
                offset = self._parse_line_offset(line, current_line, index_path.name)
                if offset is None:
                    continue

                # Skip already processed lines
                if current_line <= start_line:
                    prev_offset = offset
                    continue

                yield from self._process_chunk(dump_file, dump_path.name, prev_offset, offset)
                prev_offset = offset

                if current_line % self._progress_flush_interval == 0:
                    self._save_progress(index_path, current_line)

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

    def _process_chunk(
        self, dump_file, dump_name: str, prev_offset: int | None, offset: int
    ) -> Iterator[WikipediaItem]:
        """Decompress and parse a chunk of the dump file."""
        if prev_offset is None:
            prev_offset = offset

        if prev_offset == offset:
            return

        length = offset - prev_offset
        dump_file.seek(prev_offset)
        data = dump_file.read(length)

        try:
            decompressed = bz2.decompress(data)
            xml_content = decompressed.decode("utf-8", errors="ignore")
            yield from self._extract_pages_from_xml(xml_content)
        except OSError as exc:
            self.logger.error(
                "Failed to decompress chunk from %s between offsets %s and %s: %s",
                dump_name, prev_offset, offset, exc
            )

    def _extract_pages_from_xml(self, xml_content: str) -> Iterator[WikipediaItem]:
        """Extract and parse all pages from XML content."""
        for page_match in re.finditer(r"<page>(.*?)</page>", xml_content, re.DOTALL):
            page_xml = page_match.group(0)
            if not self._should_ignore_page(page_xml):
                item = self._parse_page_xml(page_xml)
                if item:
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

    def _should_ignore_page(self, xml_page: str) -> bool:
        """Check if a page should be ignored based on title or type."""
        if re.search(r"<redirect\s", xml_page):
            return True
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
