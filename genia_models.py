import torch.nn as nn
import torch
import torch.nn.functional as f
from transformers import BertModel


class GENIAModel(nn.Module):

    def __init__(self, dmodel, num_categories=75, category_window_size=5, device="cuda") -> None:
        super(GENIAModel, self).__init__()

        assert(category_window_size%2 != 0), "category_window_size must be an odd number"
        self.num_categories = num_categories

        self.engine = BertModel.from_pretrained("bert-base-uncased")

        self.category_wnd_size = category_window_size
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(category_window_size, 1)
        )

        self.category_embeddings = nn.Embedding(num_categories, dmodel)

        self.conv2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(num_categories, 1)
        )


    def forward(self, input_ids, token_type_ids, attention_mask, target_labels):
        """  
        @params:
        - input_ids     =>  pytorch tensor of dimension (d,n)
                            d sentences each containing n words
        - target_labels =>  one-hot encoded pytorch tensor of dimension (d,n,num_categories)
        """

        # (d,n,768)
        engine_vector = self.engine(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        embeddings = self.engine.embedding(input_ids)
        d,n,dmodel = embeddings.shape

        surround = self.category_wnd_size//2

        # (d, n, dmodel) => (d, n + surround*2, dmodel)
        embeddings = torch.concat(
            [
                torch.zeros(d, surround, dmodel),
                embeddings,
                torch.zeros(d, surround, dmodel)
            ],
            dim =1
        )

        # calculate the number of blocks formed after grouping consecutive elements into groups of size self.category_wnd_size
        # for 10 elements in the sequence, we can have 6 5-tuple sequences (10 - (5-1))
        # e.g: [1,2,3,4,5], category_wnd_size = 3 => [ (1,2,3), (2,3,4), (3,4,5) ]
        # surround * 2 = self.category_wnd_size - 1
        # num_blocks => n
        # (d, n + surround*2, dmodel) => (d * n, category_wnd_size, dmodel)
        
        context_vectors = f.unfold(
            embeddings.view(d,1,n+surround*2,dmodel), 
            kernel_size=(self.category_wnd_size, 1)
        )

        # since we already have sliding window for each word (i.e cwnd words surrounding it),
        # we can reshape embeddings as (d*n, 1, cwnd, dmodel) 
        # this is equivalent to (N,C,H,W) for conv layer
        context_vectors = context_vectors.view(d*n, 1, self.category_wnd_size, dmodel)

        # (d*n, 1, cwnd, dmodel) => (d*n, 1, 1, dmodel) => (d,n,dmodel)
        context_vectors = f.relu(self.conv1(context_vectors)).view(d,n,dmodel)

        # generate a list of indices from 0 to num_categories
        indices = torch.FloatTensor(range(self.num_categories))

        # (num_categories) => (d,n,num_categories)
        indices = indices.repeat(d,n,1)
        
        category_vectors = torch.mul(
                                self.category_embeddings(indices), # (d,n,num_categories,dmodel)
                                target_labels.reshape(d,n,self.num_categories, 1) # (d,n,num_categories,1)
                            ).reshape(d*n, 1, self.num_categories, dmodel) #  (d,n,num_categories,dmodel) => (d*n, 1, num_categories, dmodel)

        # (d*n, 1, num_categories, dmodel) => (d*n, 1, 1, dmodel) => (d,n,dmodel)
        category_vectors = f.relu(self.conv2(category_vectors)).reshape(d,n,dmodel)

        # (d,n,dmodel) + (d,n,dmodel) => (d,n,2*dmodel)
        wagon_vector = torch.cat((context_vectors, category_vectors), dim=-1)