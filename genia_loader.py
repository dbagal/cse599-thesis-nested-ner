"""  
Installation:
pip3 install beautifulsoup4==4.10.0
pip3 install lxml==4.7.1
"""

from bs4 import BeautifulSoup
import bs4
import re, os
from tqdm import tqdm  
from torch.utils.data import Dataset
import os, re
import torch

from bert_tokenizer import BERTTokenizer


class GENIADataset(Dataset):

    def __init__(self, tokenizer:BERTTokenizer, semantic_categories_file, data_folder, max_seq_len = 256) -> None:
        """
        @params:
        - semantic_categories_file  =>  file containing names of all entities/categories separated by new line
                                        (excluding outside tag)
        """
        self.tokenizer = tokenizer

        with open(semantic_categories_file, "r") as fp:
            categories = fp.readlines()
        
        num_labels = len(categories) + 1  # 1 for the outside tag

        # store category <=> idx bidirectional mapping in self.semantic_categories
        self.semantic_categories = {i+1:label.strip("\n") for i,label in enumerate(categories)}
        self.semantic_categories.update({label.strip("\n"):i+1 for i,label in enumerate(categories)})
        self.semantic_categories.update({0:"outside", "outside":0})

        # store boundary-and-category-idx <=> bio-idx bidirectional mapping in self.category_boundaries
        self.category_boundaries = {"outside": self.semantic_categories["outside"],}
        c = 1
        for i in range(1, num_labels):
            self.category_boundaries[f"b-{self.semantic_categories[i]}"] = c
            self.category_boundaries[f"i-{self.semantic_categories[i]}"] = c + 1
            c += 2
        self.category_boundaries.update({v:k for k,v in self.category_boundaries.items()})

        self.output_vector_len = len(self.category_boundaries)//2

        # get the xml file in the data folder
        xml_file = None
        for file in os.listdir(data_folder):
            if file.endswith(".xml"):
                xml_file = file
                break
        
        xml_file = os.path.join(data_folder, xml_file)

        input_ids, target_labels = self.load(xml_file, max_seq_len)

        self.input_ids = torch.LongTensor(input_ids)
        self.target_labels = torch.FloatTensor(target_labels)

        #print(self.input_ids.shape)
        #print(self.target_labels.shape)
         
    
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

        @returns:
        - chunks        =>  2D list containing sentence chunks for all sentences in the abstract
        - categories    =>  3D list containing multi-category labels for every token in every sentence
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

            for sent_token_labels in sent_labels:
                bio_token_labels = []

                for category_idx in sent_token_labels:
                    semantic_category = self.semantic_categories[category_idx]
                    # if category index was absent for previous token, then assign that category a 'B' tag
                    if category_idx!=0 and previously_present[category_idx] == False:
                        bio_token_labels += [self.category_boundaries[f"b-{semantic_category}"]]
                        previously_present[category_idx] = True

                    # if category index was present for previous token, then assign that category an 'I' tag
                    elif category_idx!=0 and previously_present[category_idx] == True:
                        bio_token_labels += [self.category_boundaries[f"i-{semantic_category}"]]
                        previously_present[category_idx] = True

                    elif category_idx == 0:
                        bio_token_labels += [0]

                # for categories which are absent for this token, 
                # set their previously_present values to False
                absent_category_indices = all_category_indices.difference(set(sent_token_labels))
                for category_idx in absent_category_indices:
                    previously_present[category_idx] = False

                bio_sent_labels += [bio_token_labels]

            bio_labels += [bio_sent_labels]
        
        return bio_labels


    def one_hot_encoding(self, labels):
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
        
        @returns:
        - one_hot_labels => 3D list containing one-hot vectors instead of individual indices
                            e.g: [28,7] => [0,0,...1,..........1]
        """
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


    def load(self, xml_file, max_seq_len):
        
        with open(xml_file, "r") as fp:
            content = fp.read()      

        print("\nExtracting abstracts ...")
        abstracts = BeautifulSoup(content, "lxml").find_all("abstract")

        sentences = []
        print("\nExtracting sentences from abstracts ...")
        for i in tqdm(range(len(abstracts))):
            abstract = abstracts[i]
            sentences += abstract.find_all("sentence")

        input_ids, token_labels = [], []  # 2D and 3D lists respectively
        observed_max_len = -1
        print("\nTokenizing text chunks and their respective categories from each sentence ...")
        for i in tqdm(range(len(sentences))):
            chunks, chunk_categories  = self.__process_sentence(sentences[i])
            sent_tokens = []
            sent_token_labels = []

            # tokenize every chunk in the sequence and repeat the chunk label "len(tokens)" times 
            # so that each token gets its own chunk_label 
            for i,chunk in enumerate(chunks):
                tokens = self.tokenizer.tokenize(chunk)
                sent_tokens += tokens
                sent_token_labels += [chunk_categories[i],]*len(tokens)

            # record the actual length of the max sequence
            observed_max_len = max(observed_max_len, len(sent_tokens))

            # convert the list of all words in the sentence to their corresponding indices
            input_ids.append(self.tokenizer.word_to_idx(sent_tokens))
            token_labels.append(sent_token_labels)

        max_seq_len = min(max_seq_len, observed_max_len)

        print("\nPadding input data and target labels ...")
        for i in tqdm(range(len(sentences))):
            
            o = [self.semantic_categories["outside"],]
            cls = self.tokenizer.word_to_idx(["[CLS]"])[0]
            sep = self.tokenizer.word_to_idx(["[SEP]"])[0]
            pad = self.tokenizer.word_to_idx(["[PAD]"])[0]

            n = len(input_ids[i])

            if n < max_seq_len - 2:
                # pad the sequence with pad tokens
                input_ids[i] = [ cls, *input_ids[i], sep,  *[pad,]*(max_seq_len - 2 - n) ]
                token_labels[i] = [ o, *token_labels[i], o, *[o,]*(max_seq_len - 2 - n) ]
            else:
                # truncate the sequence to max_seq_len
                input_ids[i] = [ cls, *input_ids[i][:max_seq_len-2], sep ]
                token_labels[i] = [ o, *token_labels[i][:max_seq_len-2], o ]

        print("\nConverting category labels according to BIO scheme ...")
        bio_labels = self.one_hot_encoding(self.bio_labels(token_labels))

        
        return input_ids, bio_labels


    def __len__(self):
        return self.target_labels.shape[0]


    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_labels[idx]


if __name__ == "__main__":

    tokenizer = BERTTokenizer()
    semantic_categories_file = "/Users/dhavalbagal/Documents/GitHub/cse599-thesis-nested-ner/semantic-categories.txt"
    data_folder = "/Users/dhavalbagal/Documents/GitHub/cse599-thesis-nested-ner/genia-dataset"
    d = GENIADataset(tokenizer, semantic_categories_file, data_folder)