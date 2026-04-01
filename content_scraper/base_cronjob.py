"""
Master script for Knowledge Hub content scraper cronjob.

wiki_check() checks the RSS feed of the wiki dumps for both content and index files, and if a new dump is detected,
calls the download_latest_dump() function from wiki_dump_update.py to download the latest dump.

As such, future knowledge base update functions can be created here, and the actual individual download/management
functions can be created in separate files in the scripts directory, which can be called here when needed.

The latest dump dates are stored in a latest_dump_date.json file, which is loaded at the start of the script and
updated if a new dump is detected.
"""
import json
import requests
from bs4 import BeautifulSoup
import datetime

from scripts import wiki_dump_update

# Enwiki dump RSS Feed URLs
BASE_ENWIKI_INDEX_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"
BASE_ENWIKI_CONTENT_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2-rss.xml"

# Frwiki dump RSS Feed URLs
BASE_FRWIKI_INDEX_URL = "https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"
BASE_FRWIKI_CONTENT_URL = "https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles-multistream.xml.bz2-rss.xml"

dump_date_file = "content_scraper/latest_dump_date.json"

def load_latest_dump_date():
    """
    Loads the latest dump dates from a latest_dump_date.json file. If the file does not exist, returns error
    """
    dump_dates = {}
    try:
        with open(dump_date_file, "r") as f:
            dump_dates = json.load(f)
            print(f"Loaded latest dump dates from file: {dump_dates}")
    except FileNotFoundError:
        print("No latest_dump_date.json file found. Must resolve.")
        return None
    return dump_dates

def save_latest_dump_date(dump_dates):
    """Saves the latest dump dates to the latest_dump_date.json file."""
    with open(dump_date_file, "w") as f:
        json.dump(dump_dates, f)

def wiki_check(wiki_dump_content_url, wiki_dump_index_url, dump_key, dump_dates):
    """Checks wikidump rss feed for latest dumpdate, if different, call update function and save new date to file."""
    page = requests.get(wiki_dump_index_url, verify="/etc/ssl/certs/ca-certificates.crt")
    print(f"Current timestamp: {datetime.datetime.now()}")

    latest_dump_date = dump_dates.get(dump_key)
    if latest_dump_date:
        print(f"Latest stored dump date in our records: {latest_dump_date}")
    else:
        print("No latest dump date found in our records.")
        latest_dump_date = ""

    soup = BeautifulSoup(page.content, "xml")
    published_date = soup.find("pubDate")
    string_published_date = str(published_date.string)

    if published_date and string_published_date != latest_dump_date:
        print(f"Link published date from the RSS feed: {string_published_date} is different from latest recorded dump date: {latest_dump_date}")
        dump_dates[dump_key] = string_published_date
        wiki_dump_update.download_latest_dump(wiki_dump_content_url, wiki_dump_index_url)
    else:
        print("No new dump detected.")

if __name__ == "__main__":
    dump_dates = load_latest_dump_date()

    print("Starting enwiki check...")
    wiki_check(BASE_ENWIKI_CONTENT_URL, BASE_ENWIKI_INDEX_URL, "enwiki", dump_dates)

    print("==========================================================")

    print("Starting frwiki check...")
    wiki_check(BASE_FRWIKI_CONTENT_URL, BASE_FRWIKI_INDEX_URL, "frwiki", dump_dates)

    save_latest_dump_date(dump_dates)
