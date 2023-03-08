# imports
import os
import sys
import argparse
import csv
from tkinter.font import names
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoConfig, AutoTokenizer


input_file = sys.srgv[1]

output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
#defining tokenizer
tokenizer = Automodel.from_pretrained("allenai/biomed_roberta_base")

#defining model
model = AutoModel.from_pretrained("allenai/biomed_roberta_base")

def biomed_roberta_embeddings(df, tokenizer, model, text_list):

    def extract_embeddings(tokenizer, model, text):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        last_hidden_state = output[0]
        cls_embeddings = last_hidden_state[:, 0, :].detach()

        return cls_embeddings.numpy().flatten().tolist()

    df['embeddings'] = None
    df['embeddings'] = df['description'].apply(extract_embeddings(tokenizer, model, df['description']))

    return df

df = pd.read_csv(input_file, columns = 'description')

outputs = biomed_roberta_embeddings(tokenizer, model, 'Hi Hardwaller, Like you I am in a similar situation. My partner was diagnosed 6 years ago, last 2 years her cognitive abilities have decreased significantly to the point where a conversation that lasts more than 3 mins ultimately ends up in a loop of repetition, struggles with words, and for the last 6 months has had issues with her eyes to the point where she can no longer read and struggles to make out people\'s face in photographs. Like your wife, my partner doesn\'t let it bother her (within the family unit/home), according to the Neuro sufferers don\'t actually realise they have a problem. She does accept that she has a small problem with her memory god bless her. She does find going out quite hard as she\'s afraid people will find her "stupid" which is very heartbreaking as she used to be fiercely independent. But we muddle by still trying to have a fulfilling life, we also have children (19/14/9) although oldest is off on his own. I wonder out of interest was your partner ever on Gilenya?')


if __name__ == "__main__":
    file_path = sys.argv[1]
    output_path = sys.argv[2]

    dout = biomed_roberta[extract_embeddings(file_path)

    dout.to_csv(output_path)
