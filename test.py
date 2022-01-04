from os import set_inheritable
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
xml_file = "./genia-raw-dataset/GENIAcorpus3.02.xml"

loader = GENIADataLoader(semantic_categories_file)
loader.load(xml_file, "./genia-raw-dataset")