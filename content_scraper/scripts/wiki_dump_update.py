"""
Script to manage download of either French or English wiki dumps,
through downloading both index and content files, which will be stored in the content/content_storage directory.
"""

import os

import requests
from bs4 import BeautifulSoup
import hashlib
import logging

DOWNLOAD_DIRECTORY = "content/content_storage"

logger = logging.getLogger(__name__)

def download_latest_dump(wiki_dump_content_url, wiki_dump_index_url):
    """Downloads the latest dump from the given wiki dump content and index URLs."""

    logger.info("============Downloading latest dump...============")

    list = list_maker(wiki_dump_content_url, wiki_dump_index_url)
    logger.info(f"List of download links retrieved: {list}")

    for url in list:
        logger.info(f"Processing URL: {url}")

        # Since the URLs in the RSS feed may start with http, we need to replace it with https and also change the
        # domain from download.wikimedia.org to dumps.wikimedia.org to successfully download the files (to fix redirect issue)
        final_url = url.replace("http://download.wikimedia.org", "https://dumps.wikimedia.org")

        filename = final_url.split("/")[-1]
        logger.info(f"Extracted filename: {filename}")

        response = requests.get(final_url, stream = True, verify="/etc/ssl/certs/ca-certificates.crt")
        if response.status_code == 200:
            with open(f"{DOWNLOAD_DIRECTORY}/{filename}", "wb") as f:
                logger.info(f"Saving to {DOWNLOAD_DIRECTORY}/{filename}...")
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded {filename} to {DOWNLOAD_DIRECTORY}, launching checksum verification...")
            checksum_verification(filename)
        else:
            logger.error(f"Failed to download {filename}. HTTP status code: {response.status_code}")

def list_maker(wiki_dump_content_url, wiki_dump_index_url):
    """Helper function to extract download links from the content and index RSS feeds."""

    link_list = []

    content_file = requests.get(wiki_dump_content_url, verify = "/etc/ssl/certs/ca-certificates.crt")
    index_file = requests.get(wiki_dump_index_url, verify = "/etc/ssl/certs/ca-certificates.crt")

    content_soup = BeautifulSoup(content_file.content, "xml")
    index_soup = BeautifulSoup(index_file.content, "xml")

    content_download = content_soup.find("item").find("description")
    index_download = index_soup.find("item").find("description")

    contentsoup = BeautifulSoup(content_download.string, "html.parser")
    indexsoup = BeautifulSoup(index_download.string, "html.parser")

    content_download_a = contentsoup.find("a", href=True).get("href")
    index_download_a = indexsoup.find("a", href=True).get("href")

    # For testing, the content file is very large, so you may comment out the below line to only download the index file
    link_list.append(content_download_a)
    link_list.append(index_download_a)
    return link_list

def checksum_verification(filename):
    """Placeholder function for checksum verification of downloaded files."""
    dump_Date = filename.split("-")[1]
    dump_lang = filename.split("-")[0]

    md5_url = f"https://dumps.wikimedia.org/{dump_lang}/{dump_Date}/{dump_lang}-{dump_Date}-md5sums.txt"
    logger.info(f"Constructed MD5 URL: {md5_url}")

    page = requests.get(md5_url, verify="/etc/ssl/certs/ca-certificates.crt")
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, "html.parser")
        md5_text = soup.get_text()

        # Find the line in the MD5 text that corresponds to our filename (failed if returns -1)
        find_filename = md5_text.find(filename)

        if find_filename == -1:
            logger.error(f"Filename {filename} not found in MD5 checksums, check to ensure it exists/ensure that dump was not",
                  "aborted (https://dumps.wikimedia.org/backup-index.html). Deleting file...")
            os.remove(f"{DOWNLOAD_DIRECTORY}/{filename}")
            return

        # Extract the MD5 hash from the line that contains our filename
        md5_text_parsed = next((line for line in md5_text.split("\n") if filename in line), None)
        md5_string = md5_text_parsed.split()[0] if md5_text_parsed else None

        hashing_verification_md5(filename, md5_string)

def hashing_verification_md5(filename, good_hash):
    """Placeholder function for hashing verification of downloaded files."""
    hash_md5 = hashlib.new('md5')

    # Generate the MD5 hash of the downloaded file
    with open(f"{DOWNLOAD_DIRECTORY}/{filename}", "rb") as f:
        while chunk := f.read(8192):
            hash_md5.update(chunk)

    calculated_hash = hash_md5.hexdigest()

    if calculated_hash == good_hash:
        logger.info(f"MD5 hash verification successful for {filename}.")
    else:
        logger.error(f"MD5 hash verification failed for {filename}. Expected: {good_hash}, Got: {calculated_hash}, deleting file...")
        os.remove(f"{DOWNLOAD_DIRECTORY}/{filename}")