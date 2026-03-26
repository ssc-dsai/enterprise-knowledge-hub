import requests
from bs4 import BeautifulSoup
import os
import sys

# ALSO DO THE LOGIC FOR FRENCH ONE- DOWNLOAD TOA  SEPARATE FOLDER MAYBE CONTENT STORAGE OR SOMETHING
# create env flag!!
from base_cronjob import BASE_ENWIKI_INDEX_URL, BASE_ENWIKI_CONTENT_URL

print(os.getcwd())
print(sys.path)
# DOWNLOAD_DIRECTORY = "./"
#ABOVE WORKSS try below tommorow
DOWNLOAD_DIRECTORY = "content/content_storage"

def download_latest_dump():
    print("============Downloading latest dump...============")

    list = list_maker()
    print(f"List of download links: {list}")

    for url in list:
        new_url = "https://dumps.wikimedia.org/enwiki/20260301/enwiki-20260301-pages-articles-multistream-index.txt.bz2"
        print(f"Downloading from URL: {new_url}")
        filename = new_url.split("/")[-1]
        print(f"Extracted filename: {filename}")
        response = requests.get(new_url, stream = True, verify="/etc/ssl/certs/ca-certificates.crt")
        if response.status_code == 200:
            with open(f"{DOWNLOAD_DIRECTORY}/{filename}", "wb") as f:
                print(f"Saving to {DOWNLOAD_DIRECTORY}/{filename}...")
                for chunk in response.iter_content(chunk_size=8192):
                    # print(f"Writing chunk of size {len(chunk)} bytes...")
                    f.write(chunk)
            print(f"Successfully downloaded {filename} to {DOWNLOAD_DIRECTORY}")
        else:
            print(f"Failed to download {filename}. HTTP status code: {response.status_code}")

# test using script only
# Add simplicity notice- no need to use loops if this just works
def list_maker():
    link_list = []

    content_file = requests.get(BASE_ENWIKI_CONTENT_URL, verify = "/etc/ssl/certs/ca-certificates.crt")
    index_file = requests.get(BASE_ENWIKI_INDEX_URL, verify = "/etc/ssl/certs/ca-certificates.crt")

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