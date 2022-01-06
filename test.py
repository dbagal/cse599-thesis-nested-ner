from bs4 import BeautifulSoup
import re
from genia_loader import *


xml = """
<sentence>Using <cons lex="(AND biochemical_analysis mutagenesis_analysis)" sem="(AND G#other_name G#other_name)"><cons lex="biochemical*">biochemical <cons sem="G#protein_molecule"> just testing </cons></cons> and <cons lex="mutagenesis*">mutagenesis</cons> <cons lex="*analysis">analyses</cons></cons>, we determined that multiple <cons lex="nuclear_factor" sem="G#protein_family_or_group">nuclear factors</cons> bind to these independent sites.</sentence>
"""
sentences = BeautifulSoup(xml, "lxml").find_all("sentence")

def tokenizer(text):
    word_pattern = re.compile(r'''([0-9]+|[-/,\[\]{}`~!@#$%\^&*()_\+=:;"'?<>])|(\.) |(\.$)|([a-z]'[a-z])| ''')
    tokens = [token for token in word_pattern.split(text) if token]
    return tokens

semantic_categories_file = "semantic-categories.txt"
xml_file = "./genia-dataset/GENIAcorpus3.02.xml"

loader = GENIAPreprocessor(semantic_categories_file)
loader.load(xml_file, tokenizer, "./genia-dataset")

# print(loader.bio_label_indices)

""" labels = [
    [[0], [28,7], [28,7], [28,], [3]],
    [[0], [0], [2,7], [0,], [2]]
]

bio = loader.bio_labels(labels)
print(bio)
catnames = loader.bio_label_indices

for sent in bio:
    for chunk in sent:
        c_labels = [catnames[i] for i in chunk]
        print(c_labels, end=",")

    print() """

""" sent_chunks = [
    ["activation of the", "human cells", "lead to"],
    ["hi how are", "you, ", "i am just", "testing"]
]

sent_chunk_categories = [
    [[0], [31,3], [31]],
    [[1,6], [6], [5,6], [4]]
]

for i in tqdm(range(len(sent_chunks))):
    sent_tokens = []
    token_indices = []
    for j in range(len(sent_chunks[i])):
        tokens = tokenizer(sent_chunks[i][j])
        sent_tokens += tokens
        token_indices += [sent_chunk_categories[i][j],]*len(tokens)
    sent_chunks[i] = sent_tokens
    sent_chunk_categories[i] = token_indices

for i in range(len(sent_chunks)):
    print(f"sent: {sent_chunks[i]}")
    print(f"labels: {sent_chunk_categories}") """