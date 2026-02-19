"""Unit tests for Wikipedia _process_index_file behavior."""
# pylint: disable=protected-access,unused-argument

import bz2
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from services.knowledge.wikipedia import WikipediaKnowedgeService

INDEX_FILE_CONTENT = """5:1:Foo1
5:2:Foo2
5:3:Foo3
10:4:Bar1
10:5:Bar1
10:6:Bar1
10:7:Bar1
111:13:Baz
111:16:Baz0
111:17:Bazzino
"""

class TestWikipediaProcessIndexFile(unittest.TestCase):
    """mock harness for testing the index process for wikipedia"""
    def _build_service(self) -> WikipediaKnowedgeService:
        repository = MagicMock()
        repository.update_history_table_start.return_value = 123
        logger = MagicMock()
        queue_service = MagicMock()

        return WikipediaKnowedgeService(
            queue_service=queue_service,
            logger=logger,
            repository=repository,
        )

    @staticmethod
    def _write_bz2_text(path: Path, content: str) -> None:
        with bz2.open(path, mode="wt", encoding="utf-8") as index_file:
            index_file.write(content)

    def test_index_process(self) -> None:
        """testing to ensure we are not SKIPPING byte chunks from the end of the file speceifically."""
        service = self._build_service()
        service._progress_flush_interval = 1

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            index_path = tmp_path / "enwiki-index.txt.bz2"
            dump_path = tmp_path / "dump.xml.bz2"

            self._write_bz2_text(index_path, INDEX_FILE_CONTENT)
            dump_path.write_bytes(b"dummy")

            service._load_progress = MagicMock(return_value=0)
            service._save_progress = MagicMock()

            def mock_process_chunk(_dump_file, _dump_name, prev_offset, offset, _source):
                return [f"{prev_offset}-{offset}"]

            service._process_chunk = MagicMock(side_effect=mock_process_chunk)

            items = list(service._process_index_file(index_path=index_path, dump_path=dump_path))

            self.assertEqual(items, ["5-10", "10-111", "111-None"])

            chunk_calls = [
                (call.args[2], call.args[3])
                for call in service._process_chunk.call_args_list
            ]
            self.assertEqual(chunk_calls, [(5, 10), (10, 111), (111, None)])
            self.assertEqual(service._save_progress.call_args_list[-1].args[1], 10)


if __name__ == "__main__":
    unittest.main()
