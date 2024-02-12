import logging
import random
import time
import urllib
import urllib.error
import urllib.request
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import json

def get_snarxiv_titles(base_url) -> list:
    """
    Returns 5 random snarxiv titles from the snarxiv homepage.
    args
        base_url: snarxiv url to make the request
    """

    # hardcoded protection against too many calls
    time.sleep(5)

    response = requests.get(base_url)
    if 200 <= response.status_code < 300:
        soup = BeautifulSoup(response.text, "html.parser")
        titles = [
            title.text.replace("\n", "").replace("Title: ", "").strip()
            for title in soup.select(".list-title")
        ]
        return titles
    else:
        logging.warning(
            f"Status code {response.status_code} received, returning emptly list"
        )
        return []


def get_arxiv_titles(max_results: int = 40*10**3, num_titles: int = 10) -> list:

    """
    Returns a snapshot of num_titles at a random page of the hep arxiv category
    args
        max_results: largest query call
        num_titles: how many titles to gather at once

    Thanks to arXiv for their API functionality
    """

    # hardcoded protection against too many calls
    time.sleep(5)

    start_title = random.randint(0, max_results)
    url = f"http://export.arxiv.org/api/query?search_query=cat:hep-th&start={start_title}&max_results={num_titles}"
    xml_response = urllib.request.urlopen(url)

    if 200 <= xml_response.getcode() <= 300:
        soup = BeautifulSoup(xml_response, features="xml")
        titles = [
            entry.find("title").text.replace("\n", "").strip()
            for entry in soup.find_all("entry")
        ]
        return titles
    else:
        logging.warning(
            f"Status code {xml_response.getcode()} received, returning emptly list"
        )
        return []
    

 
def check_identical_elements(train_set: str, test_set: str):
    common_elements = set.intersection(set(train_set), set(test_set))
    return len(common_elements) == 0


def get_titles(data_save_path: str , sample_size: int , base_url_snarxiv: str):

    """
    Gets the arxiv and snarxiv titles and then saves them in json lists
            args
            base_url: snarxiv url to make the request
    """

    arxiv_titles = []
    snarxiv_titles = []

    "Get titles"
    logging.info("Getting arXiv titles")
    logging.info("*" * 100)
    while len(arxiv_titles) < sample_size:
        logging.info(f"arXiv titles gathered {len(arxiv_titles)}/{sample_size}")
        arxiv_titles=arxiv_titles+get_arxiv_titles()
        arxiv_titles=list(set(arxiv_titles))
    arxiv_titles=arxiv_titles[0:sample_size]

    logging.info("Getting snarXiv titles")
    logging.info("*" * 100)
    while len(snarxiv_titles) < sample_size:
        logging.info(f"snarXiv titles gathered {len(snarxiv_titles)}/{sample_size}")
        snarxiv_titles = snarxiv_titles+get_snarxiv_titles(base_url_snarxiv)
        snarxiv_titles=list(set(snarxiv_titles))
    snarxiv_titles=snarxiv_titles[0:sample_size]
    
    "Split lists into train and test sets"
    arxiv_titles_train, arxiv_titles_test, snarxiv_titles_train, snarxiv_titles_test = train_test_split(arxiv_titles, snarxiv_titles, test_size=0.2, random_state=42)

    if check_identical_elements(arxiv_titles_train, arxiv_titles_test) and check_identical_elements(snarxiv_titles_train, snarxiv_titles_test):
        logging.info("Checked that there are no identical elements between train and test sets.")
    else:
        logging.info("Train and test sets have some identical elements. Please handle accordingly.")

    "Saving"
    with open(f"{data_save_path}/snarxiv_titles_train", "w") as f:
        json.dump(snarxiv_titles_train, f)
    with open(f"{data_save_path}/snarxiv_titles_test", "w") as f:
        json.dump(snarxiv_titles_test, f)
    with open(f"{data_save_path}/arxiv_titles_train", "w") as f:
        json.dump(arxiv_titles_train, f)
    with open(f"{data_save_path}/arxiv_titles_test", "w") as f:
        json.dump(arxiv_titles_test, f)

 
