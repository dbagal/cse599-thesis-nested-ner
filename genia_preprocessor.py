from bs4 import BeautifulSoup
from bs4.element import Tag
from torch.utils.data import Dataset
import re, os
import json


class GENIADatasetPreprocessor():

    def __init__(self, tokenizer:object, label_file) -> None:
        
        self.tokenizer = tokenizer
        with open(label_file, "r") as fp:
            labels = fp.readlines()

        self.label_to_idx = {label.strip(" ").strip("\n"):i for i, label in enumerate(labels)}


    def create_batches(self, xml_file, batch_size, output_dir):
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n<?xml-stylesheet type="text/css" href="gpml.css" ?>\n<!DOCTYPE set SYSTEM "gpml.dtd">\n<set>\n<import resource="GENIAontology.daml" prefix="G"></import>\n'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(xml_file, "r") as fp:
            content = fp.read().replace("\n", " ").replace("\t", " ")      

        bs_xml = BeautifulSoup(content, "lxml")
        abstracts = bs_xml.find_all("abstract")

        xml_name = xml_file.split("/")[-1].strip(".xml")

        for i in range(0, len(abstracts), batch_size):
            data_file_name = os.path.join(output_dir, xml_name+"#"+str(i)+"-"+str(i+batch_size-1)+".xml")
            with open(data_file_name, "w") as fp:
                batched_abstracts = [str(tag) for tag in abstracts[i:i+batch_size]]
                batched_abstracts = "\n".join(batched_abstracts)
                fp.write(xml+batched_abstracts+"\n</set>")


    def _get_tokens(self, tag, tokens=[]):
        for child in tag.children:
            if type(child) is Tag:
                self.get_string(child, tokens)
            else:
                tokens.append(child.get_text())

    
    def _get_outermost_label(self, tag):
        label_pattern = re.compile(r'G#([-\w_#@$%&/+=]+)\s*')

        if tag.name == "cons":
            sem_attrib_value = tag.get("sem", None)
            if sem_attrib_value is None: labels = [self.label_to_idx["O"]]
            else: labels = [self.label_to_idx[label] for label in label_pattern.findall(sem_attrib_value)]

        else:
            labels = [self.label_to_idx["O"]]

        return labels


    def _process(self, xml, stack=[]):
        tokens = []
        annotations = []
        stack += self._get_outermost_label(xml)

        for child in xml.children:
            if type(child) is Tag:
                child_tokens, child_annotations, stack = self._process(child, stack)
                tokens += child_tokens
                annotations += child_annotations
                stack = stack[:-1]
            else:
                tokens.append(child)
                annotation = list(set(stack.copy()))
                if self.label_to_idx["O"] in stack and len(stack)>1:
                    annotation.remove(self.label_to_idx["O"])
                annotations.append(annotation)

        return tokens, annotations, stack


    def _prepare_xml(self, xml_file, output_dir):
        
        with open(xml_file, "r") as fp:
            content = fp.read()      

        bs_xml = BeautifulSoup(content, "lxml")
        abstracts = bs_xml.find_all("abstract")

        sentences = []
        for abstract in abstracts:
            sentences += abstract.find_all("sentence")

        input_seq_chunks, labels = [], []
        for sent in sentences:
            stack = []
            textlist, annotations, _  = self._process(sent, stack)
            input_seq_chunks.append(textlist)
            labels.append(annotations)

        input_tokens = []
        output_labels = []
        for i,input_seq in enumerate(input_seq_chunks):
            seq_tokens = []
            seq_labels = []
            for j,chunk in enumerate(input_seq):
                chunk_tokens = self.tokenizer(chunk)
                seq_tokens.extend(chunk_tokens)
                label = labels[i][j]
                seq_labels.extend([label,]*len(chunk_tokens))
            input_tokens.append(seq_tokens)
            output_labels.append(seq_labels)

        json_data = {
            "inputs": input_tokens,
            "outputs": output_labels
        }

        xml_name = xml_file.split("/")[-1].strip(".xml")
        jsonfile = os.path.join(output_dir, xml_name+".json")

        with open(jsonfile, "w") as fp:
            json.dump(json_data, fp)

      
    def prepare_dataset(self, data_folder, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        xml_files = [file_name for file_name in os.listdir(data_folder) if file_name.endswith(".xml")]

        for xml_file in xml_files:
            self._prepare_xml(os.path.join(data_folder, xml_file), output_dir)
            #print(f"File '{xml_file}' processed!")


if __name__=="__main__":

    def tokenizer(text):
        word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?<>])|(\.) |(\.$)|([a-z]'[a-z])| ''')
        tokens = [token for token in word_pattern.split(text) if token]
        return tokens

    xmlfile = '/content/cse599-thesis/data/GENIAcorpus3.02.xml'
    data_folder = "/content/cse599-thesis/data/genia-dataset/"
    output_dir = "/content/cse599-thesis/data/genia-dataset-processed/"
    label_file = '/content/cse599-thesis/data/labels.txt'

    loader = GENIADatasetPreprocessor(tokenizer, label_file)

    loader.create_batches(xmlfile, 1, "/content/cse599-thesis/data/genia-dataset/")
    loader.prepare_dataset(data_folder, output_dir)