from tokenizers import Tokenizer
from transformers import BertTokenizer

class GENIABERTTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # you can access BERT tokenizer keys as => self.tokenizer.vocab.keys()

    
    def __call__(self, text):
        """  
        @params:
        @returns:
        - a dict of the form: {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        """
        return self.tokenizer.encode_plus(
                        sent,                           # Sentence to encode.
                        add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                        max_length = None,
                        truncation = True,              # Pad & truncate all sentences.
                        padding = 'longest',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',          # Return pytorch tensors.
                    )


    def word_to_idx(self, token):
        # for bert, UNK has index 100
        return self.tokenizer.convert_tokens_to_ids(token)

        


    def idx_to_word(self, idx):
        return self.tokenizer.decode([idx])
"""  
>>> tokenizer.decode([ 101, 7632, 1010, 2129, 2024, 2017, 1029,  102])
'[CLS] hi, how are you? [SEP]'
"""