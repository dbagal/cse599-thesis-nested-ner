import torch
from torch.utils.data import DataLoader
import re, os
from collections import defaultdict
from models import GENIAReformer

import warnings
warnings.filterwarnings("ignore")

from genia_dataloader import *
from vocab import *


class GENIAEval():

    def __init__(self, vocab_size, vocab_file_path, label_file, model_path, device) -> None:
        BATCH_SIZE = 256

        self.vocab_size = vocab_size
        self.vocab_file_path = vocab_file_path
        self.label_file = label_file
        self.model_path = model_path

        self.PAD_IDX = 0
        self.UNK_IDX = 1

        self.vocab  = Vocab()
        self.vocab.load(self.vocab_file_path)

        self.device=device

        # initialise the model
        self.model = GENIAReformer(
            dmodel=1024, 
            dqk=512, 
            dv=512, 
            heads=4, 
            feedforward=2048,
            vocab_size=self.vocab_size, 
            num_buckets=32, 
            num_bio_labels=77, 
            device=self.device
        )

        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.idx_to_label = {0: 'O', 1: 'B_amino_acid_monomer', 2: 'I_amino_acid_monomer', 3: 'B_peptide', 4: 'I_peptide', 5: 'B_protein_N/A', 
                            6: 'I_protein_N/A', 7: 'B_protein_complex', 8: 'I_protein_complex', 9: 'B_protein_domain_or_region', 10: 'I_protein_domain_or_region', 
                            11: 'B_protein_family_or_group', 12: 'I_protein_family_or_group', 13: 'B_protein_molecule', 14: 'I_protein_molecule', 15: 'B_protein_substructure', 
                            16: 'I_protein_substructure', 17: 'B_protein_subunit', 18: 'I_protein_subunit', 19: 'B_nucleotide', 20: 'I_nucleotide', 
                            21: 'B_polynucleotide', 22: 'I_polynucleotide', 23: 'B_DNA_N/A', 24: 'I_DNA_N/A', 25: 'B_DNA_domain_or_region', 
                            26: 'I_DNA_domain_or_region', 27: 'B_DNA_family_or_group', 28: 'I_DNA_family_or_group', 29: 'B_DNA_molecule', 30: 'I_DNA_molecule', 
                            31: 'B_DNA_substructure', 32: 'I_DNA_substructure', 33: 'B_RNA_N/A', 34: 'I_RNA_N/A', 35: 'B_RNA_domain_or_region', 
                            36: 'I_RNA_domain_or_region', 37: 'B_RNA_family_or_group', 38: 'I_RNA_family_or_group', 39: 'B_RNA_molecule', 40: 'I_RNA_molecule', 
                            41: 'B_RNA_substructure', 42: 'I_RNA_substructure', 43: 'B_other_organic_compound', 44: 'I_other_organic_compound', 45: 'B_organic', 
                            46: 'I_organic', 47: 'B_inorganic', 48: 'I_inorganic', 49: 'B_atom', 50: 'I_atom', 
                            51: 'B_carbohydrate', 52: 'I_carbohydrate', 53: 'B_lipid', 54: 'I_lipid', 55: 'B_virus', 
                            56: 'I_virus', 57: 'B_mono_cell', 58: 'I_mono_cell', 59: 'B_multi_cell', 60: 'I_multi_cell', 
                            61: 'B_body_part', 62: 'I_body_part', 63: 'B_tissue', 64: 'I_tissue', 65: 'B_cell_type', 
                            66: 'I_cell_type', 67: 'B_cell_component', 68: 'I_cell_component', 69: 'B_cell_line', 70: 'I_cell_line', 
                            71: 'B_other_artificial_source', 72: 'I_other_artificial_source', 73: 'B_other_name', 74: 'I_other_name', 75: 'B_coordinated', 
                            76: 'I_coordinated'}


    @staticmethod
    def tokenizer(text):
        word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?<>])|(\.) |(\.$)|([a-z]'[a-z])| ''')
        tokens = [token for token in word_pattern.split(text) if token]
        return tokens


    def eval(self, sents, max_sent_len, labels=None):
        tokenized_sents = [GENIAEval.tokenizer(sent) for sent in sents] # (d,n)

        _input = [
            [self.vocab.word_to_idx.get(token, self.UNK_IDX) for token in GENIAEval.tokenizer(sent)] 
            for sent in sents
        ]

        if max_sent_len is None:
            max_sent_len = len(max(_input, key=len))

        for i,sent in enumerate(_input):
            if len(sent) < max_sent_len:       
                _input[i] += [self.PAD_IDX,]*(max_sent_len - len(_input[i])) 
            elif len(sent) > max_sent_len:
                _input[i] = _input[i][0:max_sent_len]

        _input_tensors = torch.LongTensor(_input).to(self.device) # (d,n)
        output = self.model(_input_tensors) # (d,n,num_categories)
        output[output>0.5] = 1
        output[output<=0.5] = 0 
        
        temp = output.view(-1)
        #print((temp == 1).nonzero(as_tuple=True)[0])
        if labels is not None:
          output = labels

        processed_output = []
        for example in output:
            sent = []
            for token in example:
                labels = []
                for i,label in enumerate(token):
                    if label==1.0:
                        labels += [self.idx_to_label[i]]
                sent += [labels]
            processed_output += [sent]
        
        entities = defaultdict(list)
        for i,sent in enumerate(tokenized_sents):
            for j,token in enumerate(sent):
                labels = processed_output[i][j]
                for label in labels:
                    txt_label = "_".join(label.split("_")[1:])
                    if label.startswith("B"):
                        entities[txt_label] += [[token]]
                    elif label.startswith("I"):
                        entities[txt_label][-1] += [token]
                        

        for ent, ent_spans in entities.items():
            print("="*(len(ent)+1))
            print(ent)
            print("="*(len(ent)+1))
            for span in ent_spans:
                entity_tokens = " ".join(span)
                print(f"{entity_tokens}")
            print("\n")

