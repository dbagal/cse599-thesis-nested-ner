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

class GENIADataLoader:

    def __init__(self, semantic_categories_file) -> None:
        
        with open(semantic_categories_file, "r") as fp:
            categories = fp.readlines()

        self.categories = {i+1:label.strip("\n") for i,label in enumerate(categories)}
        self.categories.update({label.strip("\n"):i+1 for i,label in enumerate(categories)})
        self.categories.update({0:"outside", "outside":0})
        
    
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
            category_indices.add(self.categories[coordinated_category])
        
        # if this 'cons' element does NOT contain coordinating structures inside, 
        # its category will be determined by its OWN SEM attribute
        elif sem_attrib_val is not None and "AND" not in sem_attrib_val:
            category = category_pattern.search(sem_attrib_val).group(1)
            category_indices.add(self.categories[category])
        
        # default category is 'outside';
        # it is only applied when this 'cons' element doesn't belong to any of the 37 genia categories
        else:
            category_indices.add(self.categories["outside"])

        category_indices.update(parent_categories)

        # if this 'cons' element is associated with more than 1 categories, 
        # then remove the 'outside' category if present as it is only applied when this 'cons' element 
        # doesn't belong to any of the 37 genia categories
        if len(category_indices)>1:
            category_indices = category_indices.difference({self.categories["outside"]})

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
                categories += [[self.categories["outside"]]]

            # for cons element, get its text chunks and associated multi-category labels and 
            # extend the 'chunks' and 'categories' lists
            elif type(sent_item)==bs4.element.Tag and sent_item.name == "cons":
                text_chunks, chunk_categories = self.__process_cons_element(sent_item)
                chunks += text_chunks
                categories += chunk_categories
        
        return chunks, categories
                  

    def load(self, xml_file, output_dir=""):
        
        with open(xml_file, "r") as fp:
            content = fp.read()      

        abstracts = BeautifulSoup(content, "lxml").find_all("abstract")

        sentences = []
        for i in tqdm(range(len(abstracts))):
            abstract = abstracts[i]
            sentences += abstract.find_all("sentence")

        sent_chunks, sent_chunk_categories = [], []
        for i in tqdm(range(len(sentences))):
            chunks, chunk_categories  = self.__process_sentence(sentences[i])
            sent_chunks += [chunks]
            sent_chunk_categories += [chunk_categories]

        category_mapping = {i:self.categories[i] for i in range(len(self.categories)//2)}

        json_data = {
            "categories":category_mapping,
            "inputs": sent_chunks,
            "outputs": sent_chunk_categories
        }

        xml_name = xml_file.split("/")[-1].strip(".xml")
        jsonfile = os.path.join(output_dir, xml_name+".json")

        with open(jsonfile, "w") as fp:
            json.dump(json_data, fp)