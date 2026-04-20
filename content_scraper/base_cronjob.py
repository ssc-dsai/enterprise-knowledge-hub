"""
Master script for Knowledge Hub content scraper cronjob (WIKI SPECIFIC AT THE MOMENT)

main() calls wiki_rss_feed_check(), which checks the RSS feed of the wiki dumps for both content and index files, and
if a new dump is detected, calls the wiki_download_latest_dump() function to download the latest dump.

At the moment, everything is done in one file, but in the future,
knowledge base update functions can be created here, and the actual individual download/management
functions can be created in separate files a separate scripts directory, which can be called here when needed.

The latest dump dates are stored in the run history table

TO TEST USING PSQL, we insert logs with a different dump_date in the metadata field, and check if the cronjob detects
the new dump (triggered through a fastapi-crons endpoint) and updates the log accordingly:

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
import os
import hashlib
import requests
from bs4 import BeautifulSoup

from services.database.run_history_service import RunHistoryService
from services.knowledge.models import RunStatus

DOWNLOAD_DIRECTORY = "content/content_storage"

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

ENABLE_SSL_VERIFICATION = os.getenv("ENABLE_SSL_VERIFICATION", "True")

logger = logging.getLogger(__name__)
_run_history_service = RunHistoryService(logger)

def wiki_download_latest_dump(wiki_dump_content_url, wiki_dump_index_url):
    """Downloads the latest dump from the given wiki dump content and index URLs."""

    logger.info("============Downloading latest dump...============")

    list_of_links = wiki_list_maker(wiki_dump_content_url, wiki_dump_index_url)
    logger.info("List of download links retrieved: %s", list_of_links)

    for url in list_of_links:
        logger.info("Processing URL: %s", url)

        # Since the URLs in the RSS feed may start with http, we need to replace it with https and also change the
        # domain from download.wikimedia.org to dumps.wikimedia.org to successfully download the files
        # (to fix redirect issue)
        final_url = url.replace("http://download.wikimedia.org", "https://dumps.wikimedia.org")

        filename = final_url.split("/")[-1]
        logger.info("Extracted filename: %s", filename)

        response = requests.get(final_url, stream = True, verify=ENABLE_SSL_VERIFICATION,
                                timeout=30)
        if response.status_code == 200:
            with open(f"{DOWNLOAD_DIRECTORY}/{filename}", "wb") as f:
                logger.info("Saving to %s/%s...", DOWNLOAD_DIRECTORY, filename)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Successfully downloaded %s to %s, launching checksum verification...",
                        filename, DOWNLOAD_DIRECTORY)
            wiki_checksum_verification_hash_extractor(filename)
        else:
            logger.error("Failed to download %s. HTTP status code: %s", filename, response.status_code)

def wiki_list_maker(wiki_dump_content_url, wiki_dump_index_url):
    """Helper function to extract download links from the content and index RSS feeds."""

    link_list = []

    content_file = requests.get(wiki_dump_content_url, verify=ENABLE_SSL_VERIFICATION,
                                timeout=30)
    index_file = requests.get(wiki_dump_index_url, verify=ENABLE_SSL_VERIFICATION, timeout=30)

    content_soup = BeautifulSoup(content_file.content, "xml")
    index_soup = BeautifulSoup(index_file.content, "xml")

    content_download = content_soup.find("item").find("description")
    index_download = index_soup.find("item").find("description")

    contentsoup = BeautifulSoup(content_download.string, "html.parser")
    indexsoup = BeautifulSoup(index_download.string, "html.parser")

    content_download_a = contentsoup.find("a", href=True).get("href")
    index_download_a = indexsoup.find("a", href=True).get("href")

    link_list.append(content_download_a)
    link_list.append(index_download_a)
    return link_list

def wiki_checksum_verification_hash_extractor(filename):
    """Function for retrieving confirmed hashes of downloaded files."""
    dump_date = filename.split("-")[1]
    dump_lang = filename.split("-")[0]

    md5_url = f"https://dumps.wikimedia.org/{dump_lang}/{dump_date}/{dump_lang}-{dump_date}-md5sums.txt"
    logger.info("Constructed MD5 URL: %s", md5_url)

    page = requests.get(md5_url, verify=ENABLE_SSL_VERIFICATION, timeout=30)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, "html.parser")
        md5_text = soup.get_text()

        # Find the line in the MD5 text that corresponds to our filename (failed if returns -1)
        find_filename = md5_text.find(filename)

        if find_filename == -1:
            logger.error("Filename %s not found in MD5 checksums, check to ensure it exists/ensure that",
                         filename)
            os.remove(f"{DOWNLOAD_DIRECTORY}/{filename}")
            return

        # Extract the MD5 hash from the line that contains our filename
        md5_text_parsed = next((line for line in md5_text.split("\n") if filename in line), None)
        md5_string = md5_text_parsed.split()[0] if md5_text_parsed else None

        wiki_hashing_verification_md5(filename, md5_string)

def wiki_hashing_verification_md5(filename, good_hash):
    """Function for hashing verification (creation through downloaded file and comparison with
    confirmed hashes) of downloaded files."""
    hash_md5 = hashlib.new('md5')

    # Generate the MD5 hash of the downloaded file
    with open(f"{DOWNLOAD_DIRECTORY}/{filename}", "rb") as f:
        while chunk := f.read(8192):
            hash_md5.update(chunk)

    calculated_hash = hash_md5.hexdigest()

    if calculated_hash == good_hash:
        logger.info("MD5 hash verification successful for %s.", filename)
    else:
        logger.error("MD5 hash verification failed for %s. Expected: %s, Got: %s, deleting file...",
                     filename, good_hash, calculated_hash)
        os.remove(f"{DOWNLOAD_DIRECTORY}/{filename}")

def wiki_rss_feed_check(wiki_dump_content_url, wiki_dump_index_url, dump_key):
    """Checks wikidump rss feed for latest dumpdate, if different, call update function and save new date to file."""
    page = requests.get(wiki_dump_index_url, verify=ENABLE_SSL_VERIFICATION, timeout=30)

    logger.info("Current timestamp: %s", datetime.datetime.now())

    latest_dump_date = _run_history_service.cronjob_get_most_recent_dump_date("cronjob-" + dump_key)

    if latest_dump_date:
        logger.info("Latest stored dump date in our records: %s", latest_dump_date)
    else:
        logger.info("No latest dump date found in our records.")
        latest_dump_date = ""

    soup = BeautifulSoup(page.content, "xml")
    published_date = soup.find("pubDate")
    string_published_date = str(published_date.string)

    if published_date and string_published_date != latest_dump_date:
        logger.info("Link published date from the RSS feed: %s is different from latest recorded dump date: %s",
                     string_published_date, latest_dump_date)

        wiki_download_latest_dump(wiki_dump_content_url, wiki_dump_index_url)
        logger.info("inserting log into DB")
        _run_history_service.cronjob_insert_new_log("cronjob-" + dump_key, RunStatus.DUMP_LINK_UPDATED,
                                                   {"dump_date": string_published_date}, datetime.datetime.now())
    else:
        logger.info("No new dump detected.")

def main():
    """Main function to run the KB checks, which will be run as a cronjob cyclically"""
    logger.info("Starting enwiki check...")
    wiki_rss_feed_check(BASE_ENWIKI_CONTENT_URL, BASE_ENWIKI_INDEX_URL, "enwiki")

    logger.info("==========================================================")

    logger.info("Starting frwiki check...")
    wiki_rss_feed_check(BASE_FRWIKI_CONTENT_URL, BASE_FRWIKI_INDEX_URL, "frwiki")
