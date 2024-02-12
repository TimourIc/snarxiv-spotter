from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from src.train import save_joint_dataset_json 
from src.test import manual_test
from src.utils import set_logger
import os
import yaml
import json

"""TESTING THE MODEL --- TO MAKE SURE THAT WE HAVE MADE NO MISTAKE LET US PERFORM A CRUDE AND MANUAL TEST ON A NEVER-SEEN TEST SET"""

logger=set_logger(os.path.basename(__file__)[0:-3])

#model params:
MODEL_NAME= 'distilbert-base-uncased'

#config_params:
with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
data_path = config_data["paths"]["data_path"]
results_path= config_data["paths"]["results_path"]

#loading model
model_path=f"{results_path}/models/{MODEL_NAME}"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(model_path) 

#loading test set
save_joint_dataset_json(data_path , mode="test")
with open(f"{data_path}/titles_test.json") as file:
    titles_test=json.load(file)

if __name__=="__main__":
    
    accuracy=manual_test(tokenizer, model, titles_test)
    print(f"Accuracy of model: {accuracy}")
    with open(f"{data_path}/accuracy_test_set.json", "w") as f:
        json.dump(accuracy, f)

