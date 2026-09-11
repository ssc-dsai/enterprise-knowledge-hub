"""Unit tests for TBS policies page-fetching logic (isolated from DB, queue, embeddings).

Run live tests (real HTTP calls to TBS) with:
    uv run -m pytest tests/services/knowledge/test_tbs_policies_fetch.py -v -s -k live
"""
# pylint: disable=protected-access

import enum
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Python 3.10 compat: backfill StrEnum if missing
if not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        """Minimal StrEnum backfill for Python < 3.11."""
    enum.StrEnum = _StrEnum

# Stub out heavy/unavailable modules before importing the service
_stub = MagicMock()
for _mod in (
    "playhouse", "playhouse.postgres_ext",
    "peewee",
    "pgvector", "pgvector.peewee",
    "repository.database", "repository.base_model",
    "repository.knowledge_tbs_policies_model", "repository.knowledge_tbs_policies",
    "services.database.tbs_policy_item_service",
    "services.database.run_history_service",
    "services.queue.queue_worker", "services.queue.queue_service",
    "provider.embedding.qwen3.embedder_factory",
    "torch",
):
    sys.modules.setdefault(_mod, MagicMock())

from services.knowledge.tbs_policies.tbs_policies import TBSPoliciesKnowledgeService  # pylint: disable=wrong-import-position


# ── Sample HTML fixtures ──────────────────────────────────────────────────────

HIERARCHY_HTML = """
<html><body>
<ul>
  <li><div class="tv-top"><a id="id12345" href="doc-eng.aspx?id=12345">Policy A</a></div></li>
  <li><div class="tv-top"><a id="id67890" href="doc-eng.aspx?id=67890">Policy B</a></div></li>
  <li><div class="tv-top"><a id="id11111" href="doc-eng.aspx?id=11111">Directive C</a></div></li>
  <li><div class="tv-top"><a id="id12345" href="doc-eng.aspx?id=12345">Policy A (duplicate)</a></div></li>
  <li><a id="notanid">Should be ignored</a></li>
</ul>
</body></html>
"""

# Mirrors the real TBS page structure: <main class="container"> containing a sidebar
# (pol-sb) and the numbered policy body in <div id="ps-doc">.
POLICY_PAGE_HTML = """
<html><head>
  <meta name="dcterms.modified" content="2025-06-15"/>
</head><body>
<main class="container">
  <h1 id="wb-cont" property="name">Policy on Service and Digital</h1>
  <section class="pol-sb">
    <h2>Supporting tools</h2>
    <ul><li><a href="#">Sidebar link — should not appear in extracted content</a></li></ul>
  </section>
  <div id="ps-doc">
    <div class="pol-content">
      <p>This policy outlines the requirements for the management of service delivery,
         information, data, information technology, and cyber security in the Government
         of Canada. It supports a modern and digital government.</p>
      <p>Section 4.1 — The objective of this policy is to ensure that the government's
         service delivery is client-centric by design.</p>
    </div>
  </div>
</main>
<footer>Site footer</footer>
</body></html>
"""

# No dcterms.modified meta — exercises the <dt>/<dd> fallback date extraction.
POLICY_PAGE_WITH_DATE_DL = """
<html><body>
<main class="container">
  <h1>Directive on Open Government</h1>
  <div id="ps-doc">
    <p>This directive establishes requirements for the release of information and data
       under the Government of Canada's commitment to transparent operations.</p>
    <p>Departments must publish their information assets in machine-readable formats.</p>
  </div>
</main>
<dl id="wb-dtmd">
  <dt>Date modified:</dt>
  <dd><time property="dateModified">2024-03-01</time></dd>
</dl>
</body></html>
"""

# Content inside ps-doc is intentionally too short to pass the 50-char minimum.
POLICY_PAGE_MINIMAL_HTML = """
<html><body>
<main class="container">
  <h1>Short Doc</h1>
  <div id="ps-doc"><p>Too short.</p></div>
</main>
</body></html>
"""

POLICY_PAGE_NO_CONTENT = """
<html><body></body></html>
"""


class TestTBSPoliciesFetch(unittest.TestCase):
    """Tests for _fetch_page_ids and _fetch_policy_page in isolation."""

    def _build_service(self) -> TBSPoliciesKnowledgeService:
        """Create a TBSPoliciesKnowledgeService with all dependencies mocked."""
        queue_service = MagicMock()
        logger = MagicMock()
        run_history_service = MagicMock()
        run_history_service.select_first_instance_of_run_id.return_value = None

        svc = TBSPoliciesKnowledgeService(
            queue_service=queue_service,
            logger=logger,
            run_history_service=run_history_service,
        )
        # Mock the DB service so no real DB calls are made
        svc._tbs_policy_service = MagicMock()
        return svc

    # ── _fetch_page_ids ───────────────────────────────────────────────────────

    def test_fetch_page_ids_extracts_unique_ids(self):
        """Should extract unique page IDs from <li id='idXXXX'> elements."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.text = HIERARCHY_HTML
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        page_ids = svc._fetch_page_ids()

        self.assertEqual(page_ids, [12345, 67890, 11111])
        svc.session.get.assert_called_once()

    def test_fetch_page_ids_empty_page(self):
        """Should return an empty list when hierarchy page has no matching <li> elements."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.text = "<html><body><ul><li>No ID here</li></ul></body></html>"
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        page_ids = svc._fetch_page_ids()

        self.assertEqual(page_ids, [])

    # ── _fetch_policy_page ────────────────────────────────────────────────────

    def test_fetch_policy_page_parses_content_and_metadata(self):
        """Should extract title, content from ps-doc, and last modified from dcterms.modified meta."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.text = POLICY_PAGE_HTML
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        item = svc._fetch_policy_page(99999)

        self.assertIsNotNone(item)
        self.assertEqual(item.name, "Policy on Service and Digital")
        self.assertEqual(item.page_id, 99999)
        self.assertEqual(item.source, "tbs-policies")
        self.assertIn("client-centric", item.content)
        # Sidebar content must be excluded — ps-doc selector isolates the policy body
        self.assertNotIn("Sidebar link", item.content)
        # Date from <meta name="dcterms.modified">
        self.assertEqual(item.last_modified_date, datetime(2025, 6, 15))

    def test_fetch_policy_page_extracts_date_from_dl(self):
        """Should fall back to <dt>Date modified</dt>/<dd><time> when dcterms.modified meta is absent."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.text = POLICY_PAGE_WITH_DATE_DL
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        item = svc._fetch_policy_page(77777)

        self.assertIsNotNone(item)
        self.assertEqual(item.last_modified_date, datetime(2024, 3, 1))

    def test_fetch_policy_page_dcterms_meta_takes_priority_over_dt(self):
        """dcterms.modified meta must win over a conflicting <dt>/<dd> date."""
        svc = self._build_service()

        html = """
        <html><head>
          <meta name="dcterms.modified" content="2025-01-01"/>
        </head><body>
        <main class="container">
          <h1>Conflicting Dates Policy</h1>
          <div id="ps-doc">
            <p>Content that is long enough to exceed the fifty character minimum check applied
               during ingestion so this item is not skipped by the length guard.</p>
          </div>
        </main>
        <dl id="wb-dtmd">
          <dt>Date modified:</dt>
          <dd><time property="dateModified">2017-08-24</time></dd>
        </dl>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        item = svc._fetch_policy_page(88888)

        self.assertIsNotNone(item)
        # dcterms.modified (2025-01-01) must win over the dt/dd footer date (2017-08-24)
        self.assertEqual(item.last_modified_date, datetime(2025, 1, 1))

    def test_fetch_policy_page_skips_short_content(self):
        """Should return None when ps-doc content is too short (< 50 chars)."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.text = POLICY_PAGE_MINIMAL_HTML
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        item = svc._fetch_policy_page(11111)

        self.assertIsNone(item)

    def test_fetch_policy_page_no_body(self):
        """Should return None when the page has no body at all."""
        svc = self._build_service()

        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        item = svc._fetch_policy_page(00000)

        self.assertIsNone(item)

    def test_fetch_policy_page_strips_scripts_and_nav(self):
        """Should remove <script>, <style>, <nav>, <header>, <footer> from extracted content."""
        svc = self._build_service()

        html = """
        <html><body>
        <main class="container">
          <h1>Clean Policy</h1>
          <div id="ps-doc">
            <script>alert('xss')</script>
            <nav><a href="/">Home</a></nav>
            <p>This is the actual policy content that should remain visible after parsing
               and should be long enough to pass the minimum length filter easily.</p>
            <style>.hidden { display: none; }</style>
          </div>
        </main>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()
        svc.session.get = MagicMock(return_value=mock_response)

        item = svc._fetch_policy_page(55555)

        self.assertIsNotNone(item)
        self.assertNotIn("alert", item.content)
        self.assertNotIn("Home", item.content)
        self.assertNotIn(".hidden", item.content)
        self.assertIn("actual policy content", item.content)

    # ── fetch_from_source (integration of fetch + skip logic) ─────────────────

    @patch("time.sleep", return_value=None)
    def test_fetch_from_source_skips_up_to_date(self, _mock_sleep):
        """Should skip items the DB says are up-to-date and yield the rest."""
        svc = self._build_service()

        # Simulate hierarchy returning two IDs
        mock_hierarchy_resp = MagicMock()
        mock_hierarchy_resp.text = """
        <html><body><ul>
          <li><div class="tv-top"><a id="id100" href="doc-eng.aspx?id=100">Policy 100</a></div></li>
          <li><div class="tv-top"><a id="id200" href="doc-eng.aspx?id=200">Policy 200</a></div></li>
        </ul></body></html>
        """
        mock_hierarchy_resp.raise_for_status = MagicMock()

        policy_100_html = """
        <html><head><meta name="dcterms.modified" content="2025-01-01"/></head>
        <body><main class="container"><h1>Policy 100</h1>
        <p>Content for policy 100 with enough text to pass the fifty character minimum requirement.</p>
        </main>
        <dl id="wb-dtmd"><dt>Date modified:</dt><dd><time property="dateModified">2025-01-01</time></dd></dl>
        </body></html>
        """
        policy_200_html = """
        <html><head><meta name="dcterms.modified" content="2025-06-01"/></head>
        <body><main class="container"><h1>Policy 200</h1>
        <p>Content for policy 200 with enough text to pass the fifty character minimum requirement.</p>
        </main>
        <dl id="wb-dtmd"><dt>Date modified:</dt><dd><time property="dateModified">2025-06-01</time></dd></dl>
        </body></html>
        """

        mock_100_resp = MagicMock()
        mock_100_resp.text = policy_100_html
        mock_100_resp.raise_for_status = MagicMock()

        mock_200_resp = MagicMock()
        mock_200_resp.text = policy_200_html
        mock_200_resp.raise_for_status = MagicMock()

        # First call = hierarchy, second = page 100, third = page 200
        svc.session.get = MagicMock(side_effect=[mock_hierarchy_resp, mock_100_resp, mock_200_resp])

        # Mark page 100 as up-to-date, page 200 as stale
        def is_up_to_date(page_id, _source, _last_mod, _embedding_dims=None):
            return page_id == 100
        svc._tbs_policy_service.record_is_up_to_date.side_effect = is_up_to_date

        items = list(svc.fetch_from_source())

        # Only page 200 should be yielded
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].page_id, 200)
        self.assertEqual(items[0].name, "Policy 200")
        # Stale page should have its old chunks deleted
        svc._tbs_policy_service.delete_by_page_id_source.assert_called_once_with(200, "tbs-policies")


if __name__ == "__main__":
    unittest.main()


class TestTBSPoliciesLiveFetch(unittest.TestCase):
    """Live tests that hit real TBS URLs — run with `pytest -s -k live` to observe output."""

    def _build_service(self) -> TBSPoliciesKnowledgeService:
        queue_service = MagicMock()
        logger = MagicMock()
        run_history_service = MagicMock()
        svc = TBSPoliciesKnowledgeService(
            queue_service=queue_service,
            logger=logger,
            run_history_service=run_history_service,
        )
        svc._tbs_policy_service = MagicMock()
        return svc

    def test_live_fetch_hierarchy_and_first_pages(self):
        """Fetch the real TBS hierarchy, print discovered IDs, then fetch first 3 pages."""
        svc = self._build_service()

        # 1) Fetch real hierarchy
        page_ids = svc._fetch_page_ids()
        print(f"\n{'=' * 70}")
        print(f"HIERARCHY: discovered {len(page_ids)} page IDs")
        print(f"First 10 IDs: {page_ids[:10]}")
        print(f"{'=' * 70}")

        self.assertGreater(len(page_ids), 0, "Expected at least one page ID from hierarchy")

        # 2) Fetch first 3 individual policy pages
        fetched = 0
        for pid in page_ids[:3]:
            print(f"\n{'─' * 70}")
            print(f"Fetching page_id={pid} ...")
            item = svc._fetch_policy_page(pid)

            if item is None:
                print("  → returned None (skipped or empty)")
                continue

            fetched += 1
            print(f"  title          : {item.name}")
            print(f"  page_id        : {item.page_id}")
            print(f"  source         : {item.source}")
            print(f"  last_modified  : {item.last_modified_date}")
            print(f"  content length : {len(item.content)} chars")
            print(f"  content preview: {item.content[:300]}...")
            print(f"{'─' * 70}")

        self.assertGreater(fetched, 0, "Expected at least one page to parse successfully")
