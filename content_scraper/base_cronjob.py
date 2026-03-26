import json
import os
# import os

# import sys
# print("\n".join(sys.path))

import requests
from bs4 import BeautifulSoup
import datetime

# print(os.getcwd())

# because crontab executes the process in its own directory, we need to change the current working directory to the directory of this script
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory: {os.getcwd()}")

from scripts import wiki_dump_update

BASE_ENWIKI_INDEX_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2-rss.xml"
BASE_ENWIKI_CONTENT_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2-rss.xml"


dump_date_file = "content_scraper/latest_dump_date.json"

# print(os.getcwd())


def load_latest_dump_date():
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
    with open(dump_date_file, "w") as f:
        json.dump(dump_dates, f)

def enwiki_check(dump_key, dump_dates):
    page = requests.get(BASE_ENWIKI_INDEX_URL, verify="/etc/ssl/certs/ca-certificates.crt")
    print(f"timestamp: {datetime.datetime.now()}")

    latest_dump_date = dump_dates.get(dump_key)
    if latest_dump_date:
        print(f"Latest dump date from file: {latest_dump_date}")
    else:
        print("No latest dump date found in file.")
        latest_dump_date = ""

    soup = BeautifulSoup(page.content, "xml")
    published_date = soup.find("pubDate")
    string_published_date = str(published_date.string)
    print(string_published_date)
    print(latest_dump_date)
    print(f"type of published_date.string: {type(string_published_date)}, type of latest_dump_date: {type(latest_dump_date)}")
    if published_date and string_published_date != latest_dump_date:
        print(f"Published date: {string_published_date} is different from latest dump date: {latest_dump_date}")
        dump_dates[dump_key] = string_published_date
        wiki_dump_update.download_latest_dump()

    else:
        print("No new dump detected.")

if __name__ == "__main__":
    dump_dates = load_latest_dump_date()

    print("Starting enwiki check...")
    enwiki_check("enwiki", dump_dates)

    save_latest_dump_date(dump_dates)