import os
import json
from collections import OrderedDict,Counter


class Vocab():


    def __init__(self, vocab_size:int = None, tokenizer:object = None):
        """  
        @params 
        - tokenizer => tokenizes the input string and returns a list of tokens
        - vocab_size => number of words in the vocabulary 
        """
        
        self.vocab_start_index = 2   
        self.index = self.vocab_start_index
        self.size = 0
        self.word_to_idx = dict()

        # End of the line marker which will be substituted for \n in the text
        self.eol_marker = " "  

        if vocab_size is not None:    
            self.total_vocab_size = vocab_size  + self.vocab_start_index 

        if tokenizer is not None:     
            self.tokenizer = tokenizer


    def buildFromFolder(self, folder:str):
        paths = [os.path.join(folder,fname) for fname in os.listdir(folder) if fname.endswith(".txt")]
        
        contents = ""
        for path in paths:
            with open(path, "r") as fp:
                text = fp.read().lower().replace("\n", self.eol_marker)
            contents += self.eol_marker+text
        self.buildFromText(contents)


    def buildFromText(self, text:str):
        tokens = self.tokenizer(text)        
        word_freq = OrderedDict(sorted(Counter(tokens).items(), key=lambda x:x[1], reverse=True)[0:self.total_vocab_size - self.vocab_start_index])
        for word in word_freq.keys():
            if self.word_to_idx.get(word, None) is None:
                self.word_to_idx[word] = self.index
                self.index +=1
                self.size+=1


    def buildFromFile(self, path:str):
        with open(path, "r") as fp:
            text = fp.read().lower().replace("\n", self.eol_marker)
        self.buildFromText(text)

    
    def load(self, path:str):
        with open(path, "r") as fp:
            vocab = json.load(fp)
        self.word_to_idx = vocab
        vocab_len = len(vocab)
        self.index = vocab_len
        self.size = vocab_len


    def save(self, path:str):
        with open(path, "w") as fp:
            json.dump(self.word_to_idx, fp)