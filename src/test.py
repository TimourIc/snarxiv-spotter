
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from random import shuffle
import logging

def infer_probabilities(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, input:str) -> list:
    """Given an title input as a string retrurns the output probabilities the classes
    args:
        tokenizer: pretrained tokenizer for a model
        model: pretrained model
        input: string that represents a title that will be evaluated
    """
    input=tokenizer(input,  truncation=True, padding="max_length")
    input_ids=torch.tensor(input["input_ids"])
    attention_mask=torch.tensor(input["attention_mask"])
    output=F.softmax( model(input_ids=input_ids , attention_mask=attention_mask).logits , dim=1).tolist()

    return output


def manual_test(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, test_set: list)-> float:

    """A function that is written as a crude manual safe-check to test the model performance
    args:
        tokenizer: pretrained tokenizer
        model: pretrained classifier model
        test_set: a list of dictionaries where each element is a dict with two keys - text and label
    """
    arxiv_count=sum([el["label"] for el in test_set])
    snarxiv_count=len(test_set)-arxiv_count
    logging.info(f"Starting a manual test containing {arxiv_count} arxiv titles and {snarxiv_count} snarxiv titles")
    shuffle(test_set)

    correct_predictions=0
    for el in tqdm(test_set):
        prob=infer_probabilities(tokenizer,model,el["text"])
        pred=np.argmax(prob)
        label=el["label"]
        if pred==label:
            correct_predictions+=1
    
    accuracy=correct_predictions/len(test_set)

    return accuracy
 
