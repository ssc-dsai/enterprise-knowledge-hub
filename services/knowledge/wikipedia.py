import bz2
from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from services.knowledge.base import KnowledgeService

load_dotenv()

DONE_SUFFIX: str = ".done"
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
        ),

    _content_folder_path: Path = Path(os.getenv("WIKIPEDIA_CONTENT_FOLDER",
                                                 "./content/wikipedia")).expanduser().resolve()

    def __init__(self, queue_service, logger):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")

    def read(self) -> Iterator[dict[str, object]]:
        """Read data from Wikipedia index.txt.bz2 source.
        
            The content will be first entrypoint in the main .bz2 multistream file.
                Ex: 345,6789,Fruits (Read more on the doc from the README.md from content/ folder)
        """
        # Placeholder implementation for reading from Wikipedia
        for index_path in self._discover_index_files():
            self.logger.info("Reading data from Wikipedia source: %s", index_path)
            # Load already processed lines to support resuming
            completed_lines = self._load_completed_lines(index_path)
            if completed_lines:
                self.logger.info("Resuming: found %d already processed lines for %s",
                                 len(completed_lines), index_path.name)
            # Open up the index archive, and read line until you end up with a different byte offset (about 100 lines)
            # unzip the bites and for each articles (<page> items found in) send to the ingest queue
            temp_last_byteoffset: int | None = None
            with bz2.open(index_path, mode='rt') as index_file:
                for line in index_file:
                    try:
                        last_byteoffset = temp_last_byteoffset
                        offset_str, _, _ = line.strip().split(":", 2)
                        offset = int(offset_str)

                        if line in completed_lines:
                            # already processed this line, skip it
                            continue

                        # set the last byteoffset for the next iteration
                        temp_last_byteoffset = offset
                        if last_byteoffset is None:
                            last_byteoffset = offset
                            lenght = offset
                        else:
                            lenght = offset - last_byteoffset

                        if last_byteoffset != offset:
                            # if the byteoffset is different here it means we hit a multistream chunk end,
                            # we need to extract it (or die trying), and yield the resulting individual <page> elements
                            match = INDEX_FILENAME.match(index_path.name)
                            prefix = match.group("prefix")
                            chunk = match.group("chunk")
                            dump_name = f"{prefix}{chunk if chunk else ""}.xml.bz2"
                            dump_path = index_path.with_name(dump_name)
                            with open(dump_path, 'rb') as dump_file:
                                dump_file.seek(last_byteoffset)
                                data = dump_file.read(lenght)
                                try:
                                    decompressed = bz2.decompress(data)
                                    xml_content = decompressed.decode("utf-8", errors="ignore")
                                    for page_match in re.finditer(r"<page>(.*?)</page>", xml_content, re.DOTALL):
                                        page_xml = page_match.group(0)
                                        if not self._should_ignore_page(page_xml):
                                            yield {"wiki_page_xml": page_xml}
                                except Exception as exc:
                                    self.logger.error(
                                        "Failed to decompress chunk from %s between offsets %s and %s: %s",
                                        dump_name, last_byteoffset, offset, exc
                                    )
                    except ValueError:
                        self.logger.warning(
                            "Skipping malformed line in %s", index_path.name
                        )
                    finally:
                        # note this line as completed for this file, in case of restart later.
                        self._mark_line_as_completed(index_path, line)

    def _discover_index_files(self) -> Iterator[Path]:
        """
        index files will be named like so for wikipedia:
            * enwiki-20240620-pages-articles-multistream1-index.txt.bz2
            * frwiki-20240620-pages-articles-multistream1-index.txt.bz2.done
        """
        self.logger.debug("Searching for index files in ---> %s", self._content_folder_path)
        for node in sorted(self._content_folder_path.rglob("*.txt*")):
            if not node.is_file():
                continue
            match = INDEX_FILENAME.match(node.name)
            if not match:
                self.logger.debug("Skipping index file with unknown pattern: %s", node.name)
                continue
            yield node

    def _should_ignore_page(self, xml_page: str) -> bool:
        """
        This function will look at a <page> element and will exclude it
        based on it's title or the type of page it is (e.g., redirects).
        """
        # Check for redirect pages
        if re.search(r"<redirect\s", xml_page):
            return True
        # Extract title and check against ignored prefixes
        title_match = re.search(r"<title>([^<]+)</title>", xml_page)
        if title_match:
            title = title_match.group(1)
            for prefix in self._ignored_title_prefixes:
                if title.startswith(prefix):
                    return True
        return False

    def _mark_line_as_completed(self, index_path: Path, line: str) -> None:
        """
        Mark a line as completed by appending it to a .done file.
        """
        done_path = index_path.with_suffix(index_path.suffix + DONE_SUFFIX)
        with done_path.open('a') as done_file:
            done_file.write(line)

    def _load_completed_lines(self, index_path: Path) -> set[str]:
        """
        Load the set of already completed lines from the .done file.
        Returns an empty set if no .done file exists.
        """
        done_path = index_path.with_suffix(index_path.suffix + DONE_SUFFIX)
        if not done_path.exists():
            return set()

        with done_path.open('r') as done_file:
            return set(done_file.readlines())
