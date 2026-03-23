from time import strptime

import requests
from bs4 import BeautifulSoup
import datetime

BASE_URL = "https://dumps.wikimedia.org/enwiki/"
# add injection from env file for the below verify statement
page = requests.get(BASE_URL, verify="/etc/ssl/certs/ca-certificates.crt")

print(f"timestamp: {datetime.datetime.now()}")
print(f"Fetching data from {BASE_URL}...")

soup = BeautifulSoup(page.content, 'html.parser')
print(f"Page title: {soup.title.string}")
pre = soup.find('pre')
if pre:
    print(f"Found pre tag: {pre}")
latest_dump = pre.find("a", href = "latest/")

print(f"Latest dump: {latest_dump}")

formatted_timestamp = latest_dump.next_sibling.strip().split(' ')[0]
# print(f"timestamp: {latest_dump.next_sibling.strip()}")
print(f"timestamp: {formatted_timestamp}")

dated_timestamp = strptime(formatted_timestamp, "%d-%b-%Y")
print(f"Formatted timestamp: {dated_timestamp}")

latest_dump_url = BASE_URL + latest_dump['href']
print(f"Latest dump URL: {latest_dump_url}")