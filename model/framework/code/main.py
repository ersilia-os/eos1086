# imports
import os
import csv
import sys
import pandas as pd
import numpy as np
import transformers
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
from bs4 import BeautifulSoup
import requests
import random
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoConfig, AutoTokenizer

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

class Settings:
    batch_size=16
    max_len=350
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 318

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    
set_seed(Settings.seed)

class TrainValidDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.text = df["description"].values
        # self.target = df["target"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        texts = self.text[idx]
        tokenized = self.tokenizer.encode_plus(texts, truncation=True, add_special_tokens=True,
                                               max_length=self.max_len, padding="max_length")
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]
        # targets = self.target[idx]
        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            # "targets": torch.tensor(targets, dtype=torch.float32)
        }

#defining model
model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

class BiomedRoBERTa(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.biomed_roberta = model
        
    def forward(self, ids, mask):
        output = self.biomed_roberta(ids, attention_mask=mask)
        return output

model.to(Settings.device)

#defining tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

# extract embeddings
def extract_embeddings(path):

    response = requests.get(path)
    bs = BeautifulSoup(response.text, ['xml'])

    obs = bs.find_all("Obs")
    #<Obs OBS_CONF="F" OBS_STATUS="A" OBS_VALUE="10.7255" TIME_PERIOD="2005-04-01"/>

    df = pd.DataFrame(columns=['text'])

    for node in obs:
        df = df.append({'text': node.get("description")}, ignore_index=True)


    df = pd.read_xml(path)

    df_train = df  #As we are not learning any loss function, we are not computing 

    train_dataset = TrainValidDataset(df_train, tokenizer, Settings.max_len)
    train_loader = DataLoader(train_dataset, batch_size=Settings.batch_size,
                          shuffle=True, num_workers=8, pin_memory=True)

    
    # make mini batch data

    batch = next(iter(train_loader))

    ids = batch["ids"].to(Settings.device)
    mask = batch["mask"].to(Settings.device)

    output = model(ids, mask)

    # last_hidden_state
    last_hidden_state = output[0]

    #cls_embeddings (one method to get last hidden state representations)

    cls_embeddings = last_hidden_state[:, 0, :].detach()

    output_df = pd.DataFrame(cls_embeddings.numpy())

    final_df = pd.concate([df, output_df], axis = 1)

    return final_df


if __name__ == "__main__":
    file_path = sys.argv[1]
    output_path = sys.argv[2]

    dout = extract_embeddings(file_path)

    dout.to_csv(output_path)
