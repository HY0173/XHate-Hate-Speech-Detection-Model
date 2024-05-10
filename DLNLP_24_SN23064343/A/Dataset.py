import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# Prepare Dataset (with defined tokenizer and cleaned data)
class MyDataset(Dataset):
    def __init__(self, data_csv, tokenizer):
        self.df = data_csv
        self.tokenizer = tokenizer

    #Get (input_ids, token_type_ids, attention_mask, and label) for Training
    def __getitem__(self, index):
        data = self.df.iloc[index]
        tweet,label = data['cleaned'],data['class']
        tokenzied_dict = self.tokenizer.encode_plus(tweet,
                                                    max_length=64,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids = torch.tensor(tokenzied_dict['input_ids'])
        token_type_ids = torch.tensor(tokenzied_dict['token_type_ids'])
        attention_mask = torch.tensor(tokenzied_dict['attention_mask'])
        return input_ids,token_type_ids,attention_mask,torch.tensor(label)

    def __len__(self):
        return len(self.df)


