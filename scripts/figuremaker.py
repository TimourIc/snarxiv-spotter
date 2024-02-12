
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np

#config_params:
with open("config/config.yaml", "r") as file:
    config_data = yaml.safe_load(file)
data_path = config_data["paths"]["data_path"]
results_path= config_data["paths"]["results_path"]


MODEL_NAME= 'distilbert-base-uncased'
training_logs_path=f"{results_path}/models/{MODEL_NAME}/training_logs.json"
fig_save_path=f"{results_path}/figures"

class FigureMaker():

    def __init__(self, training_logs_path: str):

        with open(training_logs_path) as file:
            training_logs=json.load(file)
        
        self.training_logs=training_logs
    
    def process_logs(self):
        
        self.train_steps=[el["step"] for el in self.training_logs if "loss" in el.keys()]
        self.val_steps=[el["step"] for el in self.training_logs if "eval_loss" in el.keys()]
        self.train_loss=[el["loss"] for el in self.training_logs if "loss" in el.keys()]
        self.val_loss=[el["eval_loss"] for el in self.training_logs if "eval_loss" in el.keys()]
        self.val_accuracy=[el["eval_accuracy"] for el in self.training_logs if "eval_loss" in el.keys()]

    def create_fig(self, fig_save_path , fig_name):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(self.train_steps, self.train_loss, label="train loss")
        ax1.plot(self.val_steps, self.val_loss, label="val loss", linestyle="dashed", linewidth=3)
        ax1.legend()
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax2.plot(self.val_steps,self.val_accuracy, label="val accuracy", color="orange" , linestyle="dashed", linewidth=3)
        ax2.legend()
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Accuracy")
        plt.savefig(f'{fig_save_path}/{fig_name}')



if __name__=="__main__":

    figuremaker=FigureMaker(training_logs_path) 
    figuremaker.process_logs()
    figuremaker.create_fig(fig_save_path, "train_logs.png")

