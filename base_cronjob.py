"""
Master script for Knowledge Hub content scraper cronjob.

wiki_check() checks the RSS feed of the wiki dumps for both content and index files, and if a new dump is detected,
calls the download_latest_dump() function from wiki_dump_update.py to download the latest dump.

As such, future knowledge base update functions can be created here, and the actual individual download/management
functions can be created in separate files in the scripts directory, which can be called here when needed.

The latest dump dates are stored in a latest_dump_date.json file, which is loaded at the start of the script and
updated if a new dump is detected.

TO TEST USING PSQL, we insert logs with a different dump_date in the metadata field, and check if the cronjob detects
the new dump and updates the log accordingly:

=============================(SET PROPER TIMEZONE FIRST)========================

SET TIMEZONE = 'America/Toronto';

INSERT INTO run_history (service_name, status, metadata, timestamp)
VALUES
('cronjob-frwiki', 'New Dump Link Detected and Downloaded', '{"dump_date": "Wed, 07 Apr 2040 20:29:35 GMT"}', NOW());

INSERT INTO run_history (service_name, status, metadata, timestamp)
VALUES
('cronjob-enwiki', 'New Dump Link Detected and Downloaded', '{"dump_date": "Wed, 07 Apr 2040 20:29:35 GMT"}', NOW());

"""

import datetime
import logging
import requests
from bs4 import BeautifulSoup

from repository.postgrespg import WikipediaPgRepository
from services.knowledge.models import RunStatus
from content_scraper.scripts import wiki_dump_update

# Enwiki dump RSS Feed URLs
BASE_ENWIKI_INDEX_URL = (
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"
)
BASE_ENWIKI_CONTENT_URL = (
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2-rss.xml"
)

# Frwiki dump RSS Feed URLs
BASE_FRWIKI_INDEX_URL = (
    "https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"
)
BASE_FRWIKI_CONTENT_URL = (
    "https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles-multistream.xml.bz2-rss.xml"
)

run_history_repository =  WikipediaPgRepository.from_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)

def wiki_check(wiki_dump_content_url, wiki_dump_index_url, dump_key):
    """Checks wikidump rss feed for latest dumpdate, if different, call update function and save new date to file."""
    page = requests.get(wiki_dump_index_url, verify="/etc/ssl/certs/ca-certificates.crt", timeout=30)
    logging.info("Current timestamp: %s", datetime.datetime.now())

    latest_dump_date = run_history_repository.cronjob_get_most_recent_dump_date("cronjob-" + dump_key)
    logging.info("Latest dump date from DB: %s", latest_dump_date)

    if latest_dump_date:
        logging.info("Latest stored dump date in our records: %s", latest_dump_date)
    else:
        logging.info("No latest dump date found in our records.")
        latest_dump_date = ""

    soup = BeautifulSoup(page.content, "xml")
    published_date = soup.find("pubDate")
    string_published_date = str(published_date.string)

    if published_date and string_published_date != latest_dump_date:
        logging.info("Link published date from the RSS feed: %s is different from latest recorded dump date: %s",
                     string_published_date, latest_dump_date)

        wiki_dump_update.download_latest_dump(wiki_dump_content_url, wiki_dump_index_url)

        logging.info("inserting into DB")
        run_history_repository.cronjob_insert_new_log("cronjob-" + dump_key, RunStatus.DUMP_LINK_UPDATED,
                                                      {"dump_date": string_published_date}, datetime.datetime.now())

    else:
        logging.info("No new dump detected.")

if __name__ == "__main__":
    logging.info("Starting enwiki check...")
    wiki_check(BASE_ENWIKI_CONTENT_URL, BASE_ENWIKI_INDEX_URL, "enwiki")

    logging.info("==========================================================")

    logging.info("Starting frwiki check...")
    wiki_check(BASE_FRWIKI_CONTENT_URL, BASE_FRWIKI_INDEX_URL, "frwiki")
