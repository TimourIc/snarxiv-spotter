from transformers import TrainingArguments, DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.utils import set_logger
from src.train import train
import os
import yaml
import argparse

"""SCRIPT THAT FINE-TUNES A PRE-TRAINED MODEL FOR CLASSIFICATION"""

#PARAMS

#config_params:
with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
data_path = config_data["paths"]["data_path"]
results_path= config_data["paths"]["results_path"]

#model params:
MODEL_NAME= 'distilbert-base-uncased'
TOKENIZER = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
MODEL= DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

#training default params
EPOCHS=1
BATCH_SIZE=16

 
"""Set logger and load config"""
logger=set_logger(os.path.basename(__file__)[0:-3])
 

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Fine-tuning HuggingFace model")
    parser.add_argument(
    "--EPOCHS", default= EPOCHS, type=int, help=f"Number of epochs during training"
    )
    parser.add_argument(
    "--BATCH_SIZE", default= BATCH_SIZE, type=int, help=f"Batch size during training"
    )

    args = parser.parse_args()

    #Define tokenizer, model and set up training args
    training_args = TrainingArguments(
        output_dir=f'{results_path}/models/{MODEL_NAME}',        
        num_train_epochs=args.EPOCHS,               
        per_device_train_batch_size=args.BATCH_SIZE,   
        save_steps=10_000,                
        save_total_limit=2,               
        logging_strategy="steps",
        evaluation_strategy="steps",
        logging_steps=1,
        eval_steps=20,
    )

    train(
        tokenizer=TOKENIZER,
        model=MODEL,
        training_args= training_args,
        model_name= MODEL_NAME,
        data_path = data_path,
        results_path= results_path
    )

 