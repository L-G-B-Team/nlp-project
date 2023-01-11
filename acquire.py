"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked

"""
import os
import json
import pandas as pd
import time
from typing import Dict, List, Optional, Union, cast
import requests
from bs4 import BeautifulSoup

from env import github_token, github_username

REPOS_CSV = 'repos.csv'
READMES_CSV = 'readmes.csv'


def get_repo_urls() -> pd.Series:
    # TODO Docstring
    if os.path.isfile(REPOS_CSV):
        return pd.read_csv(REPOS_CSV)['0']
    all_links = []
    for p in range(1, 151):
        response = requests.get(
            f'https://github.com/search?p={p}'
            '&q=stars%3A%3E0&s=stars&type=Repositories').content
        bs = BeautifulSoup(response, 'html.parser')
        all_links += [link['href']
                      for link in bs.find_all('a', class_='v-align-middle')]
        time.sleep(5)
    lonks = pd.Series(all_links).drop_duplicates()
    lonks.to_csv(REPOS_CSV, index=False)

    return lonks


REPOS = get_repo_urls().to_list()

headers = {"Authorization": f"token {github_token}",
           "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO"
        "in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            "Error response from github api!"
            f"status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(
                    json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url},"
        f"instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url},"
        f"instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> pd.DataFrame:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


def acquire_readmes() -> pd.DataFrame:
    # TODO docstring
    if os.path.exists(READMES_CSV):
        return pd.read_csv(READMES_CSV, index_col=0)
    readme_df = pd.DataFrame(scrape_github_data())
    readme_df.to_csv(READMES_CSV)
    return readme_df
