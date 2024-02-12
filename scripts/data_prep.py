import logging
import yaml
from src.data_prep import get_titles
import argparse
import sys
from src.utils import set_logger
import os

"""SCRIPT THAT GETS THE ARXIV AND SNARXIV TITLES AND SAVES AS JSON LISTS"""

"""Set logger and load config"""
logger=set_logger(os.path.basename(__file__)[0:-3])
 


with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)

data_save_path = config_data["paths"]["data_path"]
base_url_snarxiv = config_data["urls"]["base_url_snarxiv"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparing title data")
    parser.add_argument(
    "--SAMPLE_SIZE", type=int, help=f"Number of titles to get for both classes"
    )
    args = parser.parse_args()
    get_titles(data_save_path=data_save_path,
         sample_size=args.SAMPLE_SIZE,
         base_url_snarxiv=base_url_snarxiv)
