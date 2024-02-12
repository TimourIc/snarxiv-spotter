from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from datasets import load_dataset
import logging
import numpy as np
import json
import os
import torch
import torch.nn.functional as F
from datasets import load_metric

def read_titles(data_save_path: str, mode: str = "train"):

    """Reads the titles generated after running scripts/data_prep.py
    args:
        data_save_path: location where titles are saved
        mode: read train or test set
    """

    logging.info(f"Reading saved titles from {data_save_path}")
    with open(f"{data_save_path}/arxiv_titles_{mode}") as file:
        arxiv_titles=json.load(file)
    with open(f"{data_save_path}/snarxiv_titles_{mode}") as file:
        snarxiv_titles=json.load(file)
    
    return arxiv_titles, snarxiv_titles

def save_joint_dataset_json(data_save_path: str, mode: str = "train"):

    """Creates a joint dataset of arxiv and snarxiv titles
    args:
        data_save_path: location where to save joint dataset
        name: name of joint dataset
    """

    arxiv_titles, snarxiv_titles = read_titles(data_save_path, mode)
    titles=arxiv_titles+snarxiv_titles
    labels=[1]*len(arxiv_titles)+[0]*len(snarxiv_titles)
    dataset=[{"text": title , "label": label} for title, label in zip(titles,labels) ]
    with open(f"{data_save_path}/titles_{mode}.json", "w") as f:
        json.dump(dataset, f)



def create_hf_ds_from_data(data_save_path: str , mode: str = "train"):

    """Creates a HF dataset object starting from an existing json dataset 
    args:
        data_save_path: path of dir where existing json dataset is located
    """

    dataset=load_dataset("json", data_files=f"{data_save_path}/titles_{mode}.json")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    return dataset

def prepare_dataset(data_save_path: str):
    
    """Returns HF dataset starting from title lists
    args:
        data_save_path: path of where to save intermediary steps
    """
    save_joint_dataset_json(data_save_path)
    dataset=create_hf_ds_from_data(data_save_path)
    return dataset


def compute_accuracy(eval_pred):
    """Used as a function argument in the HF trainer. Returns the accuracy of a HF model evaluation step"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return load_metric('accuracy').compute(predictions=predictions, references=labels)


def train(
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        training_args: TrainingArguments,
        model_name: str,
        data_path: str = "data",
        results_path: str = "results"
        ):
    
    """Fine-tunes a pre-trained model as a classifier for our task
    args:
        tokenizer: a HF tokenizer for a model
        model: a HF model
        training args: HF training args
        model name: HF name of the model, string used to create additional directory
        data_path: path of data directory where the titles are saved
        results_path: path of the results directory to save the model data in results_path/models/*
    """
    
    #Tokenization
    dataset=prepare_dataset(data_path)
 
    logging.info(f"Tokenizing data...")
    def preprocess_function(examples):
        return tokenizer(examples["text"],  truncation=True, padding="max_length")
    dataset=dataset.map(preprocess_function)
 
    #Training
    trainer = Trainer(
        model=model,                     
        args=training_args,              
        train_dataset=dataset["train"],   
        eval_dataset=dataset["test"],
        compute_metrics=compute_accuracy
    )
    logging.info(f"Starting Training")
    trainer.train()

    #Save training logs and model
    model_dir_path=f"{results_path}/models/{model_name}"
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    logging.info(f"Saving model to {model_dir_path}")
    trainer.save_model(model_dir_path)

    logging.info(f"Saving trainings logs to {model_dir_path}")
    with open(f"{model_dir_path}/training_logs.json", "w") as f:
        json.dump(trainer.state.log_history, f)
