# Import Dependencies
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModel
from transformers import AdamW
import warnings

from A.Dataset import MyDataset
from A.Model import HateDetector,HateDetector_CNN
from A.Train import train
from A.Test import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make path to save model
out_path = './B/result'
if not os.path.exists(out_path):
     os.makedirs(out_path)

# ==========================================Define Hyperparameters=========================================
batch_size = 32
epochs = 10
lr = 0.001
criterion = nn.CrossEntropyLoss().to(device)

# ================================================Load Data================================================
# Prepare Data for Training (8:1:1)
print("Data Splitting...")
tweets = pd.read_csv('./Datasets/clean.csv')

train_df,test_df = train_test_split(tweets,test_size=0.2,shuffle=True)
val_df,test_df = train_test_split(test_df,test_size=0.5,shuffle=True)

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
print("Loading Training Data...")
train_data = MyDataset(train_df,tokenizer)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
print("Size of Training Data: ",len(train_data))

print("Loading Validation Data...")
val_data = MyDataset(val_df,tokenizer)
val_dataloader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=True)
print("Size of Validation Data: ",len(val_data))

# Load Data for Testing
print("Loading Testing Data...")
test_data = MyDataset(test_df,tokenizer)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             shuffle=True)
print("Size of Testing Data: ",len(test_data))


# ==========================================Initialize Classifier==========================================
encoder = AutoModel.from_pretrained("vinai/bertweet-base")
print("Loading Model...")
model = HateDetector(encoder,2)
print("Classifier_1 Loaded!")

model2 = HateDetector_CNN(encoder,2)
print("Classifier_2 Loaded!")

warnings.filterwarnings('ignore')
# =========================================Model Training & Testing=========================================
# classifier 1
train(model,train_dataloader,val_dataloader,epochs,lr,criterion,out_path)
test(model,test_dataloader)

# classifier 2
train(model2,train_dataloader,val_dataloader,epochs,lr,criterion,out_path)
test(model2,test_dataloader)

print("Completed!")