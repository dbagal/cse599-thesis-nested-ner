# Installation
# !pip3 -qq install beautifulsoup4==4.10.0
# !pip3 -qq install transformers==4.16.2
# !pip3 -qq install psutil==5.9.0

from psutil import virtual_memory
ram_gb = virtual_memory().total /1073741824

print(f"Available RAM: {ram_gb}\n")

from torch.utils.data import DataLoader
import os, pickle
import warnings
warnings.filterwarnings("ignore")

from bert_tokenizer import *
from genia_loader import *
from model import *

BATCH_SIZE = 32
save_folder = "/content/gdrive/MyDrive/sbu/cse599-thesis/experiments/experiment-2/"

train_folder = "/content/gdrive/MyDrive/sbu/cse599-thesis/experiments/experiment-2/data/train"
test_folder = "/content/gdrive/MyDrive/sbu/cse599-thesis/experiments/experiment-2/data/test"
semantic_categories_file = "/content/gdrive/MyDrive/sbu/cse599-thesis/experiments/experiment-2/semantic-categories.txt"

device="cuda"
tokenizer = BERTTokenizer()

# generate and save datasets to file
 
''' train_dataset = GENIADataset(tokenizer, semantic_categories_file, train_folder)
test_dataset = GENIADataset(tokenizer, semantic_categories_file, test_folder)

with open(os.path.join(save_folder, "train-dataset.dataset"), 'wb') as fp:
  pickle.dump(train_dataset, fp)

with open(os.path.join(save_folder, "test-dataset.dataset"), 'wb') as fp:
  pickle.dump(test_dataset, fp) '''


# load train and test datasets from file
with open(os.path.join(save_folder, "train-dataset.dataset"), 'rb') as fp:
  train_dataset = pickle.load(fp)

with open(os.path.join(save_folder, "test-dataset.dataset"), 'rb') as fp:
  test_dataset = pickle.load(fp)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
train_features, train_labels = next(iter(train_loader))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_features, test_labels = next(iter(test_loader))

print(f"\nFeature batch shape: {train_features.size()}")  # BATCH_SIZE, max_sent_len
print(f"Labels batch shape: {train_labels.size()}") # BATCH_SIZE, max_sent_len, num_categories=77

print(f"\nFeature batch shape: {test_features.size()}")  # BATCH_SIZE, max_sent_len
print(f"Labels batch shape: {test_labels.size()}") # BATCH_SIZE, max_sent_len, num_categories=77

print(f"\nTraining batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

model = GENIAModel(save_folder)
model.load()

model.lr = 0.01
model.train(train_loader, test_loader, num_epochs=1)