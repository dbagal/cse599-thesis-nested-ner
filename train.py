# Installation
# !pip3 -qq install beautifulsoup4==4.10.0
# !pip3 -qq install transformers==4.16.2
# !pip3 -qq install psutil==5.9.0

from psutil import virtual_memory
ram_gb = virtual_memory().total/1073741824

print(f"Available RAM: {ram_gb} GB\n")

from torch.utils.data import DataLoader
import os, json
#import pickle
from pickle import dump, load
import warnings
warnings.filterwarnings("ignore")

from bert_tokenizer import *
from genia_loader import *
from model import *

with open(os.path.join(os.getcwd(), "train-config.json"), "r") as fp:
    config = json.load(fp)

batch_size = config["batch-size"]
main_folder = config["main-folder"]

train_file = os.path.join(main_folder, config["train-file"])
test_file = os.path.join(main_folder, config["test-file"])

device="cuda"
tokenizer = BERTTokenizer()

# generate and save datasets to file
 
train_dataset = GENIADataset(tokenizer, os.path.join(main_folder, train_file))
test_dataset = GENIADataset(tokenizer, os.path.join(main_folder, test_file))

with open(os.path.join(main_folder, "train.dataset"), 'wb') as fp:
  dump(train_dataset, fp)

with open(os.path.join(main_folder, "test.dataset"), 'wb') as fp:
  dump(test_dataset, fp)


# load train and test datasets from file
with open(os.path.join(main_folder, "train.dataset"), 'rb') as fp:
  train_dataset = load(fp)

with open(os.path.join(main_folder, "test.dataset"), 'rb') as fp:
  test_dataset = load(fp)

# define dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
train_features, train_labels = next(iter(train_loader))
test_features, test_labels = next(iter(test_loader))

print(f"\nFeature batch shape: {train_features.size()}")  # batch_size, max_sent_len
print(f"Labels batch shape: {train_labels.size()}") # batch_size, max_sent_len, num_categories=77

print(f"\nFeature batch shape: {test_features.size()}")  # batch_size, max_sent_len
print(f"Labels batch shape: {test_labels.size()}") # batch_size, max_sent_len, num_categories=77

print(f"\nTraining batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")


model = GENIAModel()
model.cuda()
#model.load()
model.lr = 0.01
model.loss_amplify_factor = 100000000
model.lr_adaptive_factor = 0.5
model.lr_patience = 3


for i in range(1):
    # first things first, backup the working model
    try:
        os.system("cp /nestedner/genia-model-v1.pt /nestedner/genia-model-v1-copy.pt")
    except:
        print("Backing up process failed")
        pass

    # train the model, save the model and calculate the losses
    train_loss, test_loss = model.train(train_loader, test_loader, num_epochs=2)
    print(f"Learning rate: {model.optimizer.param_groups[0]['lr']}\n")

    # if losses are nan, we can do error analysis and restart with the already backed-up working model
    if torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(test_loss)):
        break
    