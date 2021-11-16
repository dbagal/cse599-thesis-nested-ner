import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import re, os
from tqdm import tqdm
from models import GENIAReformer

import warnings
warnings.filterwarnings("ignore")

from genia_dataloader import *
from vocab import *

BATCH_SIZE = 256
MODEL_SAVE_PATH = "/content/model.pt"

data_folder = "/content/cse599-thesis/experiments/experiment-1/data"
vocab_size = 20000
vocab_file_path = "/content/cse599-thesis/experiments/experiment-1/vocab.json"
label_file = '/content/cse599-thesis/experiments/experiment-1/labels.txt'

vocab  = Vocab()
vocab.load(vocab_file_path)

def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?<>])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens

dataset = GENIADataset(tokenizer, vocab, label_file, data_folder, num_tokens_in_sent=None)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")  # BATCH_SIZE, max_sent_len
print(f"Labels batch shape: {train_labels.size()}") # BATCH_SIZE, max_sent_len, num_categories=77



device="cuda"

model = GENIAReformer(
    dmodel=1024, 
    dqk=512, 
    dv=512, 
    heads=4, 
    feedforward=2048,
    vocab_size=vocab_size, 
    num_buckets=32, 
    num_bio_labels=77, 
    device=device
)

#model.load_state_dict(torch.load(root+"model.pt", map_location=torch.device(device)))



optimizer = optim.Adam(model.parameters(), lr=0.01)
bce_loss = nn.BCELoss()

num_epochs = 10
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0001



model_save_path = "/content/model.pt"
progress_bar = tqdm(range(num_epochs), position=0, leave=True)

#torch.autograd.set_detect_anomaly(True)

for epoch in progress_bar:     
    epoch_loss = 0.0   
    num_batches = 0
    progress_bar.set_description(f"Epoch {epoch} ")
    
    for step, data in enumerate(train_loader):
    
        input_data, output_labels = data
        
        input_data = input_data.to(device).view(1,BATCH_SIZE, -1)
        output_labels = output_labels.to(device)
        
        optimizer.zero_grad()

        output = model(input_data)  # input_data: (1, BATCH_SIZE, max_sent_len)

        batch_loss = bce_loss(output, output_labels) 
        batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        epoch_loss += batch_loss.detach().item()
        num_batches += 1
        
        b_loss = round(epoch_loss/num_batches, 8)
        progress_bar.set_postfix({'batch':step,'batch-loss': str(b_loss)})
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    #os.system("cp /content/model.pt /content/medical-claims/model.pt") 