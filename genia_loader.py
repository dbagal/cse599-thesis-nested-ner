"""  
Installation:
pip3 install beautifulsoup4==4.10.0
pip3 install lxml==4.7.1
"""

from bs4 import BeautifulSoup
import bs4
import re, os
import json
from tqdm import tqdm  
from torch.utils.data import Dataset
import json, os, re
from vocab import Vocab
import torch


class GENIAPreprocessor:

    def __init__(self, semantic_categories_file) -> None:
        
        with open(semantic_categories_file, "r") as fp:
            categories = fp.readlines()

        self.semantic_categories = {i+1:label.strip("\n") for i,label in enumerate(categories)}
        self.semantic_categories.update({label.strip("\n"):i+1 for i,label in enumerate(categories)})
        self.semantic_categories.update({0:"outside", "outside":0})

        num_labels = len(self.semantic_categories)//2

        self.bio_label_indices = {"outside": self.semantic_categories["outside"],}
        c = 1
        for i in range(1, num_labels):
            self.bio_label_indices[f"b-{i}"] = c
            self.bio_label_indices[f"i-{i}"] = c + 1
            c += 2
        self.bio_label_indices.update({v:k for k,v in self.bio_label_indices.items()})
        
    
    def __process_cons_element(self, cons_elem, coordinated_category = None, parent_categories=set()):
        """ 
        @params:
        - cons_elem             =>  xml string of the form '<cons> .. </cons>'
        - coordinated_category  =>  coordinated category sent by the parent 'cons' to which this 'cons' element belongs to.
        - parent_categories     =>  maintains categories at every level for nested 'cons' elements
        """
        category_pattern = re.compile(r'G#([-\w_#@$%&/+=]+)\s*')
        sem_attrib_val = cons_elem.get("sem")

        category_indices = set()

        # check if this 'cons' element contains coordinated structures inside
        # if yes, the sem attribute should contain the predicate 'AND'
        # store the coordinated categories in 'coordinated_structures_inside'
        coordinated_structures_inside = None
        if sem_attrib_val is not None and "AND" in sem_attrib_val:
            coordinated_structures_inside = category_pattern.findall(sem_attrib_val)

        # if this 'cons' element is a coordinated structure, 
        # it's category will be determined by it's PARENT's SEM attribute
        if coordinated_category is not None:
            category_indices.add(self.semantic_categories[coordinated_category])
        
        # if this 'cons' element does NOT contain coordinating structures inside, 
        # its category will be determined by its OWN SEM attribute
        elif sem_attrib_val is not None and "AND" not in sem_attrib_val:
            category = category_pattern.search(sem_attrib_val).group(1)
            category_indices.add(self.semantic_categories[category])
        
        # default category is 'outside';
        # it is only applied when this 'cons' element doesn't belong to any of the 37 genia categories
        else:
            category_indices.add(self.semantic_categories["outside"])

        category_indices.update(parent_categories)

        # if this 'cons' element is associated with more than 1 categories, 
        # then remove the 'outside' category if present as it is only applied when this 'cons' element 
        # doesn't belong to any of the 37 genia categories
        if len(category_indices)>1:
            category_indices = category_indices.difference({self.semantic_categories["outside"]})

        # 1D list containing text chunks
        text_chunks = []

        # 2D list containing multi-category label for each text chunk
        chunk_categories = []

        # for every item in this 'cons' element,
        # recursively process inner 'cons' elements and for regular strings append their text chunks 
        # and the determined multi-category label ('category_indices')
        for cons_item in cons_elem.contents:

            if type(cons_item)==bs4.element.NavigableString:
                text_chunks += [cons_item]
                chunk_categories += [list(category_indices)]

            elif type(cons_item)==bs4.element.Tag and cons_item.name == "cons":
                
                # if this 'cons' element contains coordinating entities inside, then recursively process 
                # the inner entities by passing appropriate coordinated categories
                if coordinated_structures_inside is not None and len(coordinated_structures_inside)>0:
                    inner_text_chunks, inner_chunk_categories = self.__process_cons_element(
                                    cons_item, 
                                    coordinated_category=coordinated_structures_inside.pop(0), 
                                    parent_categories = category_indices
                                )
                else:
                    inner_text_chunks, inner_chunk_categories = self.__process_cons_element(
                                    cons_item, 
                                    coordinated_category = None, 
                                    parent_categories = category_indices
                                )

                text_chunks += inner_text_chunks
                chunk_categories += inner_chunk_categories

        return text_chunks, chunk_categories


    def __process_sentence(self, sentence):
        """  
        @params:
        - sentence => xml string of the form '<sentence> ... </sentence>'
        """

        # maintain a list of text chunks in the 1D list 'chunks'
        chunks = []

        # maintain multi-category label for every chunk in the 2D list 'categories'
        categories = []

        # in genia dataset, every sentence contains either a regular string or a 'cons' element
        for sent_item in sentence.contents:

            # append the regular string chunk to 'chunks' list with multi-category label as [outside]
            if type(sent_item)==bs4.element.NavigableString:
                chunks += [sent_item]
                categories += [[self.semantic_categories["outside"]]]

            # for cons element, get its text chunks and associated multi-category labels and 
            # extend the 'chunks' and 'categories' lists
            elif type(sent_item)==bs4.element.Tag and sent_item.name == "cons":
                text_chunks, chunk_categories = self.__process_cons_element(sent_item)
                chunks += text_chunks
                categories += chunk_categories
        
        return chunks, categories
                  

    def bio_labels(self, labels):
        """  
        @params:
        - labels => 3D list containing multi-category labels for every token in every sentence
                    [
                        [
                            [0,], [28,7], [74,14]
                        ],
                        [
                            ...
                        ]
                    ]
        """

        # number of semantic categories including "outside" label
        num_labels = len(self.semantic_categories)//2
        
        bio_labels = []

        # for each semantic category, we maintain information about the category if it was encountered in the previous token
        # this is required to make a decision on B, I, O tags
        previously_present = {i:False for i in range(1, num_labels)}

        # maintain a list of indices of all semantic categories excluding "outside"
        all_category_indices = set(range(1,num_labels))

        for sent_labels in tqdm(labels):
            bio_sent_labels = []

            # for every new sentence reset the previously_present meta-data
            previously_present = {i:False for i in range(1, num_labels)}

            for token_labels in sent_labels:
                bio_token_labels = []

                for category_idx in token_labels:
                    # if category index was absent for previous token, then assign that category a 'B' tag
                    if category_idx!=0 and previously_present[category_idx] == False:
                        bio_token_labels += [self.bio_label_indices[f"b-{category_idx}"]]
                        previously_present[category_idx] = True

                    # if category index was present for previous token, then assign that category an 'I' tag
                    elif category_idx!=0 and previously_present[category_idx] == True:
                        bio_token_labels += [self.bio_label_indices[f"i-{category_idx}"]]
                        previously_present[category_idx] = True

                    elif category_idx == 0:
                        bio_token_labels += [0]

                # for categories which are absent for this token, 
                # set their previously_present values to False
                absent_category_indices = all_category_indices.difference(set(token_labels))
                for category_idx in absent_category_indices:
                    previously_present[category_idx] = False

                bio_sent_labels += [bio_token_labels]

            bio_labels += [bio_sent_labels]
        
        return bio_labels


    def load(self, xml_file, tokenizer, output_dir=""):
        
        with open(xml_file, "r") as fp:
            content = fp.read()      

        print("\nExtracting abstracts ...")
        abstracts = BeautifulSoup(content, "lxml").find_all("abstract")

        sentences = []
        print("\nExtracting sentences from abstracts ...")
        for i in tqdm(range(len(abstracts))):
            abstract = abstracts[i]
            sentences += abstract.find_all("sentence")

        print("\nExtracting text chunks and their respective categories from each sentence ...")
        input_data, target_labels = [], []  # 2D and 3D lists respectively
        for i in tqdm(range(len(sentences))):
            chunks, chunk_categories  = self.__process_sentence(sentences[i])
            input_data += [chunks]
            target_labels += [chunk_categories]

        print("\nTokenizing sentence chunks ...")
        for i in tqdm(range(len(input_data))):
            sent_tokens = []
            token_indices = []
            for j in range(len(input_data[i])):
                tokens = tokenizer(input_data[i][j])
                sent_tokens += tokens
                token_indices += [target_labels[i][j],]*len(tokens)
            input_data[i] = sent_tokens
            target_labels[i] = token_indices

        print("\nConverting category labels according to BIO scheme ...")
        bio_labels = self.bio_labels(target_labels)
        
        bio_label_indices = dict()
        for k,v in self.bio_label_indices.items():
            
            if type(k)==str and k.startswith("b-"):
                bio_label_indices[v] = "b-"+self.semantic_categories[int(k.lstrip("b-"))]
            elif type(k)==str and k.startswith("i-"):
                bio_label_indices[v] = "i-"+self.semantic_categories[int(k.lstrip("i-"))]
            elif k=="outside":
                bio_label_indices[v] = k
        bio_label_indices.update({v:k for k,v in bio_label_indices.items()})

        json_data = {
            "categories":self.semantic_categories,
            "bio-categories": bio_label_indices,
            "input-data": input_data,
            "target-labels": bio_labels
        }

        xml_name = xml_file.split("/")[-1].rstrip(".xml")
        jsonfile = os.path.join(output_dir, xml_name+".json")

        with open(jsonfile, "w") as fp:
            json.dump(json_data, fp)



class GENIADataset(Dataset):
    

    def __init__(self, tokenizer:object, vocab:Vocab, semantic_categories_file, data_folder, max_seq_len = 256) -> None:
        
        super(GENIADataset, self).__init__()
        self.tokenizer = tokenizer

        self.PAD_IDX = 0
        self.UNK_IDX = 1

        self.vocab = vocab

        preprocessor = GENIAPreprocessor(semantic_categories_file)

        # get the xml file in the data folder
        xml_file = None
        for file in os.listdir(data_folder):
            if file.endswith(".xml"):
                xml_file = file
                break
        
        xml_file = os.path.join(data_folder, xml_file)
        json_file = xml_file.split("/")[-1].rstrip(".xml")+".json"
        
        # if there is no json file in the data folder, preprocess the raw genia dataset and generate the json
        if json_file not in os.listdir(data_folder):
            preprocessor.load(xml_file, tokenizer, output_dir=data_folder)

        with open(os.path.join(data_folder, json_file), "r") as fp:
            json_dataset = json.load(fp)

        # convert input tokens to their appropriate indices by looking up the vocabulary
        input_data = []
        print(f"\nLooking up token indices from vocabulary ...")
        for sent in tqdm(json_dataset["input-data"]):
            sent_token_indices = [self.vocab.word_to_idx.get(token, self.UNK_IDX) for token in sent]
            input_data.append(sent_token_indices)

        self.output_vector_len = len(json_dataset["bio-categories"].keys())//2 # 77
        self.bio_categories = json_dataset["bio-categories"]

        # convert target labels into one hot vectors
        self.input_data, self.target_labels = self.pad(
                                                    input_data, 
                                                    self.one_hot_encoding(json_dataset["target-labels"]), 
                                                    max_seq_len=max_seq_len
                                                )

        self.target_labels = torch.FloatTensor(self.target_labels)
        self.input_data = torch.LongTensor(self.input_data) 
    

    def one_hot_encoding(self, labels):
        one_hot_labels = []

        print(f"\nEncoding target labels into one-hot vectors ...")
        for seq_labels in tqdm(labels):
            one_hot_seq_labels =[]
            for token_labels in seq_labels:
                one_hot_token_label = [0,]*self.output_vector_len
                for category_idx in token_labels:
                    one_hot_token_label[category_idx] = 1
                one_hot_seq_labels.append(one_hot_token_label)
            one_hot_labels.append(one_hot_seq_labels)

        return one_hot_labels
                    

    def pad(self, input_data, target_labels, max_seq_len = None):
        
        if max_seq_len is None:
            max_seq_len = len(max(input_data, key=len))

        print(f"\nPadding input data ...")
        for i,sent in tqdm(enumerate(input_data)):
            if len(sent) < max_seq_len: 
                # pad the sentence with 0 token      
                input_data[i] += [self.PAD_IDX,]*(max_seq_len - len(input_data[i])) 
            elif len(sent) > max_seq_len:
                # truncate the sentence to 'max_seq_len' number of tokens
                input_data[i] = input_data[i][0:max_seq_len]

        print(f"\nPadding target labels ...")
        for i,sent_labels in tqdm(enumerate(target_labels)):
            if len(sent_labels) < max_seq_len:
                pad_vec = [self.PAD_IDX  for _ in range(self.output_vector_len)]
                target_labels[i] += [pad_vec for _ in range((max_seq_len - len(target_labels[i])))] 
            elif len(sent_labels) > max_seq_len:
                target_labels[i] = target_labels[i][0:max_seq_len]
                
        return input_data, target_labels 
            

    def __len__(self):
        return self.target_labels.shape[0]


    def __getitem__(self, idx):
        return self.input_data[idx], self.target_labels[idx]


if __name__=="__main":
    
    def tokenizer(text):
        word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?<>])|(\.) |(\.$)|([a-z]'[a-z])| ''')
        tokens = [token for token in word_pattern.split(text) if token]
        return tokens
        
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