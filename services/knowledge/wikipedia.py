import bz2
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from services.knowledge.base import KnowledgeService
from services.knowledge.models import WikipediaItem

load_dotenv()

PROGRESS_SUFFIX: str = ".progress"
INDEX_FILENAME = re.compile(r"(?P<prefix>.+)-index(?P<chunk>\d*)\.txt\.bz2")

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

    _progress_flush_interval: int = 1000 # for the .progress file we track line number we stpped.

    def __init__(self, queue_service, logger):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")

    def process_queue(self, knowledge_item: dict[str, object]) -> None:
        """Process ingested WikipediaItem from the queue."""
        item: WikipediaItem = WikipediaItem(**knowledge_item)
        #self.logger.debug("Processing Wikipedia item: %s", item.title)
        # add vector logic here.


    def fetch_from_source(self) -> Iterator[WikipediaItem]:
        """Read data from Wikipedia index.txt.bz2 source.

            The content will be first entrypoint in the main .bz2 multistream file.
                Ex: 345,6789,Fruits (Read more on the doc from the README.md from content/ folder)
        """
        for index_path in self._discover_index_files():
            self.logger.info("Reading data from Wikipedia source: %s", index_path)

            # Load last processed line number
            start_line = self._load_progress(index_path)
            if start_line > 0:
                self.logger.info("Resuming from line %d for %s", start_line, index_path.name)

            temp_last_byteoffset: int | None = None
            current_line = 0

            with bz2.open(index_path, mode='rt') as index_file:
                for line in index_file:
                    current_line += 1

                    # Parse the offset first
                    try:
                        offset_str, _, _ = line.strip().split(":", 2)
                        offset = int(offset_str)
                    except ValueError:
                        self.logger.warning("Skipping malformed line %d in %s", current_line, index_path.name)
                        continue

                    # Skip already processed lines
                    if current_line <= start_line:
                        temp_last_byteoffset = offset
                        continue

                    # Process the line
                    try:
                        last_byteoffset = temp_last_byteoffset
                        temp_last_byteoffset = offset

                        if last_byteoffset is None:
                            last_byteoffset = offset
                            length = offset
                        else:
                            length = offset - last_byteoffset

                        if last_byteoffset != offset:
                            match = INDEX_FILENAME.match(index_path.name)
                            prefix = match.group("prefix")
                            chunk = match.group("chunk")
                            dump_name = f"{prefix}{chunk if chunk else ''}.xml.bz2"
                            dump_path = index_path.with_name(dump_name)

                            with open(dump_path, 'rb') as dump_file:
                                dump_file.seek(last_byteoffset)
                                data = dump_file.read(length)
                                try:
                                    decompressed = bz2.decompress(data)
                                    xml_content = decompressed.decode("utf-8", errors="ignore")
                                    for page_match in re.finditer(r"<page>(.*?)</page>", xml_content, re.DOTALL):
                                        page_xml = page_match.group(0)
                                        if not self._should_ignore_page(page_xml):
                                            item = self._parse_page_xml(page_xml)
                                            if item:
                                                yield item
                                except Exception as exc:
                                    self.logger.error(
                                        "Failed to decompress chunk from %s between offsets %s and %s: %s",
                                        dump_name, last_byteoffset, offset, exc
                                    )
                    except Exception as exc:
                        self.logger.error("Error processing line %d in %s: %s", current_line, index_path.name, exc)
                    finally:
                        # Save progress periodically
                        if current_line % self._progress_flush_interval == 0:
                            self._save_progress(index_path, current_line)

                # Final progress save at end of file
                self._save_progress(index_path, current_line)
                self.logger.info("Completed %s at line %d", index_path.name, current_line)

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
        for node in sorted(self._content_folder_path.rglob("*.txt*")):
            if not node.is_file():
                continue
            if node.suffix == PROGRESS_SUFFIX:
                continue  # Skip progress files
            match = INDEX_FILENAME.match(node.name)
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
            title=title,
            content=content,
            last_modified_date=last_modified_date,
            pid=pid,
        )
