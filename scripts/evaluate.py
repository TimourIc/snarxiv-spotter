
import yaml
import argparse
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from src.test import infer_probabilities
import torch.nn.functional as F
import numpy as np 

#model params:
MODEL_NAME= 'distilbert-base-uncased'

#config_params:
with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
data_path = config_data["paths"]["data_path"]
results_path= config_data["paths"]["results_path"]

#model loading
model_path=f"{results_path}/models/{MODEL_NAME}"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(model_path) 

translator={0: "snarxiv", 1:"arxiv"}

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Evaluate a title")
    parser.add_argument(
    "--TITLE", type=str, help=f"Title for the model to evaluate"
    )
    args=parser.parse_args()
    output=infer_probabilities(tokenizer, model, args.TITLE)
    print(f"output: {output}")
    print(f"Probability for snarxiv: {output[0]}, Probability for arxiv: {output[1]}")
    print(f"Most likely source: {translator[np.argmax(output)]}")
    
