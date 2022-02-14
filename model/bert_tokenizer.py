from tokenizer import Tokenizer
from transformers import BertTokenizer

class BERTTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

        # you can access BERT tokenizer vocab words as => self.tokenizer.vocab.keys()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  
    
    def __call__(self, sequence_list:list, max_length=None, add_special_tokens=False):
        """  
        @params:
        - sequence_list =>  list of strings
        - max_length    =>  common length that all sequences should have

        @returns:
        - a dict of the form => {"input_ids": list, "token_type_ids": list, "attention_mask": list} 
                                where each list is of shape (d,n)
                                d - number of sequences
                                n - seq length

        """
        return self.tokenizer(
            sequence_list,
            add_special_tokens = add_special_tokens,      # Add '[CLS]' and '[SEP]'
            max_length = max_length,        # all sentences should have 'max_length' length
            truncation = True,              # Pad & truncate all sentences.
            padding = 'longest',
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',          # Return pytorch tensors.
        )


    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
        

    def word_to_idx(self, token_list:list):
        """ 
        @params:
        - token_list  =>  list of tokens

        @returns:
        - list of the corresponding token indices
        """
        # for bert, [UNK](100), [CLS](101), [SEP](102), [PAD](0)
        return self.tokenizer.convert_tokens_to_ids(token_list)


    def idx_to_word(self, idx_list:list):
        """  
        @params:
        - idx_list  =>  list of indices

        @returns:
        - list of the corresponding tokens
        """
        return self.tokenizer.decode([idx_list])


if __name__=="__main__":
    t = BERTTokenizer()
    print(t(["hello, how are you, i am fine", "hi my dog is cute"], max_length=None))
    # print(t(["hello, how are you, i am fine", "hi my dog is cute"], max_length=5))

    #op = t(["hello, how are you, i am fine", "hi my dog is cute"])
    #input_ids, token_type_ids, attn_mask = op["input_ids"], op["token_type_ids"], op["attention_mask"]
    #print(f"input-ids shape: {input_ids.shape}")
    #print(f"token-type-ids shape: {token_type_ids.shape}")
    #print(f"attn-mask shape: {attn_mask.shape}")
