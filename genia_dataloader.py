
from torch.utils.data import Dataset
import json, os, re
from vocab import Vocab
import torch


class GENIADataset(Dataset):
    

    def __init__(self, tokenizer:object, vocab:Vocab, label_file, data_folder, num_tokens_in_sent = 256) -> None:
        
        super(GENIADataset, self).__init__()
        self.tokenizer = tokenizer

        self.PAD_IDX = 0
        self.UNK_IDX = 1

        self.vocab = vocab

        with open(label_file, "r") as fp:
            labels = fp.readlines()

        self.label_to_idx = {label.strip(" ").strip("\n"):i for i, label in enumerate(labels)}
        self.idx_to_label = {i:label.strip(" ").strip("\n") for i, label in enumerate(labels)}
        self.bio_label_to_idx= {"O":0}

        c = 1
        for label in self.label_to_idx.keys():
            if label!="O":
                self.bio_label_to_idx["B_"+label] = c
                self.bio_label_to_idx["I_"+label] = c+1
                c+=2

        self.output_vector_len = len(self.bio_label_to_idx.keys()) # 77

        inputs, outputs = [], []
        json_files = [fname for fname in os.listdir(data_folder) if fname.endswith(".json")]
        for json_file in json_files:
            _input,  _output = self._load_dataset(os.path.join(data_folder, json_file))
            inputs += _input
            outputs += _output 

        self.inputs, self.outputs = self.pad(inputs, outputs, max_sent_len=num_tokens_in_sent)

        self.outputs = torch.FloatTensor(self.outputs)
        self.inputs = torch.LongTensor(self.inputs) 


    def pad(self, _input, _output, max_sent_len = None):
        
        if max_sent_len is None:
            max_sent_len = len(max(_input, key=len))

        for i,sent in enumerate(_input):
            if len(sent) < max_sent_len:       
                _input[i] += [self.PAD_IDX,]*(max_sent_len - len(_input[i])) 
            elif len(sent) > max_sent_len:
                _input[i] = _input[i][0:max_sent_len]

        for i,sent_labels in enumerate(_output):
            if len(sent_labels) < max_sent_len:
                pad_vec = [self.PAD_IDX  for _ in range(self.output_vector_len)]
                _output[i] += [pad_vec for _ in range((max_sent_len - len(_output[i])))] 
            elif len(sent_labels) > max_sent_len:
                _output[i] = _output[i][0:max_sent_len]
                
        return _input, _output


    def _load_dataset(self, json_file):
        with open(json_file, "r") as fp:
            content = json.load(fp)  

        input_sents_tokens = content["inputs"]
        output_labels = content["outputs"]

        _input, _output = [],[]

        """  
        _input: [
            [0,123,45,88,98]  // sentence 
            [45,78,549,1334, 546, 46]
        ]

        _output: [
            [ // sentence
                [0,1,0,1,0], // multiple labels for one word
                [0,1,0,1,1] // multiple labels for one word
            ],
            [],...
        ]
        """
        # get the index for each word and append the sentence containing word indices to the _input
        for sent in input_sents_tokens:
            sent_token_indices = [self.vocab.word_to_idx.get(token, self.UNK_IDX) for token in sent]
            _input.append(sent_token_indices)
            
                
        for sent in output_labels:
            sent_labels = []

            stack = {label:0 for label in self.label_to_idx.keys()}

            for j,token_labels_indices in enumerate(sent):

                t_label = [0,]*self.output_vector_len

                if token_labels_indices == [0,]:
                    t_label[0] = 1
                else:
                    token_labels_indices.remove(self.label_to_idx["O"])

                    for token_label_idx in token_labels_indices:
                        token_label = self.idx_to_label[token_label_idx]

                        if stack[token_label]==0:
                            label_idx = self.bio_label_to_idx["B_"+token_label]
                        else:
                            label_idx = self.bio_label_to_idx["I_"+token_label]

                        try:
                            if token_label_idx not in sent[j+1]:
                                stack[token_label] = 0
                            else:
                                stack[token_label]=1
                        except IndexError:
                            pass
                    
                        t_label[label_idx] = 1

                sent_labels += [t_label]

            _output += [sent_labels]
        
        return _input, _output     
            

    def __len__(self):
        return self.outputs.shape[0]


    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]



def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?<>])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens



""" #data_folder = "genia-dataset-processed/"
data_folder = "/Users/dhavalbagal/Library/Mobile Documents/com~apple~CloudDocs/sbu/thesis/codebase/test"
vocab_size = 20000
dataset_file = "/Users/dhavalbagal/Library/Mobile Documents/com~apple~CloudDocs/sbu/thesis/codebase/GENIAcorpus3.02.txt"
vocab_file_path = "/Users/dhavalbagal/Library/Mobile Documents/com~apple~CloudDocs/sbu/thesis/codebase/colab_supp_files/vocab.json"

#output_dir = "genia-dataset-json/"
label_file = '/Users/dhavalbagal/Library/Mobile Documents/com~apple~CloudDocs/sbu/thesis/codebase/colab_supp_files/labels.txt'
#loader = GENIADatasetPreprocessor(tokenizer, label_file)
#loader.create_batches(xmlfile, 5, "/content/cse599-thesis/data/genia-dataset/")
#print(loader.label_to_idx)
#loader.prepare_dataset(data_folder, output_dir)

## Building vocabulary from a single .txt file
#vocab = Vocab(vocab_size=vocab_size, tokenizer=tokenizer)
#vocab.buildFromFile(dataset_file)
#vocab.save(vocab_file_path)

## Loading the vocabulary
vocab  = Vocab()
vocab.load(vocab_file_path)

loader = GENIADataset(tokenizer, vocab, label_file, data_folder)
#print(len(loader[0]))

#for k,v in loader.bio_label_to_idx.items():
#    print(f"{v}: {k}") 

print({v:k for k,v in loader.bio_label_to_idx.items()}) """
