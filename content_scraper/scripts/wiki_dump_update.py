import requests
from bs4 import BeautifulSoup
from base_cronjob import BASE_ENWIKI_INDEX_URL, BASE_ENWIKI_CONTENT_URL

DOWNLOAD_DIRECTORY = "content/content_storage"

def download_latest_dump():
    print("============Downloading latest dump...============")

    list = list_maker()
    print(f"List of download links: {list}")

    for url in list:
        print(f"Processing URL: {url}")
        print(type(url))
        final_url = url.replace("http://download.wikimedia.org", "https://dumps.wikimedia.org")

        print(f"Constructed final URL: {final_url}")
        filename = final_url.split("/")[-1]
        print(f"Extracted filename: {filename}")
        response = requests.get(final_url, stream = True, verify="/etc/ssl/certs/ca-certificates.crt")
        if response.status_code == 200:
            with open(f"{DOWNLOAD_DIRECTORY}/{filename}", "wb") as f:
                print(f"Saving to {DOWNLOAD_DIRECTORY}/{filename}...")
                for chunk in response.iter_content(chunk_size=8192):
                    # print(f"Writing chunk of size {len(chunk)} bytes...")
                    f.write(chunk)
            print(f"Successfully downloaded {filename} to {DOWNLOAD_DIRECTORY}")
        else:
            print(f"Failed to download {filename}. HTTP status code: {response.status_code}")

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