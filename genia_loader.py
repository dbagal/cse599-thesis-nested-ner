from tqdm import tqdm  
from torch.utils.data import Dataset
import os, re, sys
import torch
from bert_tokenizer import BERTTokenizer


class GENIADataset(Dataset):

    def __init__(self, 
        tokenizer:BERTTokenizer, 
        data_file, 
        categories=['cell_type', 'RNA', 'DNA', 'cell_line', 'protein'], 
        max_seq_len = 64) -> None:

        self.tokenizer = tokenizer
        self.n = max_seq_len

        with open(data_file, "r") as fp:
            content = fp.read()

        c = 1
        self.bio_labels = {"O":0, 0:"O"}

        for cat in categories:
            self.bio_labels["B-"+cat] = c
            self.bio_labels["I-"+cat] = c+1
            self.bio_labels[c] = "B-"+cat
            self.bio_labels[c+1] = "I-"+cat
            c += 2

        self.nl = len(self.bio_labels)//2
        self.x, self.y = self.process(content)


    def __len__(self):
        return self.y.shape[0]


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

        
    def process(self, content):
        
        sentences = content.split("\n\n")

        xs, ys = [], []
        for sent in tqdm(sentences, desc="Sentence processing", ncols=80):
            sent = sent.split("\n")
            if len(sent) > 1:
                x,y = self.process_sentence(sent)
                xs += [x]
                ys += [y]

        pad = self.tokenizer.word_to_idx(["[PAD]"])[0]
        pad_vec = [0 for _ in range(self.nl)]

        d = len(xs)
        for i in tqdm(range(d), desc="Sentence padding", ncols=80):
            if len(xs[i]) < self.n:
                num_pad = self.n - len(xs[i])
                xs[i] += [pad,]*num_pad
                ys[i] += [pad_vec for _ in range(num_pad)]
            else:
                xs[i] = xs[i][:self.n]
                ys[i] = ys[i][:self.n]
        
        x = torch.LongTensor(xs) # (d,n)
        y = torch.FloatTensor(ys) # (d,n,nl)

        return x,y
        

    def process_sentence(self, sent:list):
        """
        @params:
        - sent   =>  text with first column as word followed by columns containing labels 
                        where each column represents labels at a particular nesting level
        """
        content = [line.split("\t") for line in sent]
        
        words, labels = zip(*[
                            (
                                line[0], 
                                [self.bio_labels[label] for label in line[1:]]
                            ) for line in content if len(line)>1
                        ]) # (n,) (n,nl)

        x = self.tokenizer.word_to_idx(words) # (n,)

        y = [[0 for _ in range(self.nl)] for _ in range(len(x))]

        for i, word_labels in enumerate(labels):
            for j in word_labels:
                y[i][j] = 1

        return x,y # (n,) (n,nl)


if __name__ == "__main__":
    tokenizer = BERTTokenizer()
    semantic_categories_file = "/Users/dhavalbagal/Documents/GitHub/cse599-thesis-nested-ner/semantic-categories.txt"
    data_folder = "/Users/dhavalbagal/Documents/GitHub/cse599-thesis-nested-ner/dataset/genia"
    d = GENIADataset(tokenizer, data_folder)
    print(d.train_x.shape, d.train_y.shape)
    print(d.test_x.shape, d.test_y.shape)
    y1 = torch.sum(d.train_y.view(-1,11), dim=0)
    y2 = torch.sum(d.test_y.view(-1,11), dim=0)
    print(y1.tolist())
    print(y2.tolist())

    


        
