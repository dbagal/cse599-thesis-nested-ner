import torch
import torch.nn as nn
import torch.nn.functional as f


def positionEncoding(d,n,dmodel, device="cpu"):

    pos = torch.arange(n)
    encoding = torch.zeros(n, dmodel) 
    power = torch.true_divide(torch.arange(0,dmodel,2), dmodel).unsqueeze(0).repeat(n,1)  # (n, dmodel/2)
    denom = torch.pow(10000, power)
    pos = pos.unsqueeze(1).repeat(1,dmodel//2)  # (n,dmodel/2)
    encoding[:,0::2] = torch.sin( torch.true_divide(pos,denom) )  # (n, dmodel/2)
    encoding[:,1::2] = torch.cos( torch.true_divide(pos,denom) )  # (n, dmodel/2)
    encoding = encoding.unsqueeze(0).repeat(d,1,1).to(device)  # (d,n,dmodel)

    return encoding



class TransformerEncoder(nn.Module):
    
    def __init__(self, dmodel, dq, dk, dv, heads, feedforward):
        super(TransformerEncoder, self).__init__()

        self.dmodel, self.dq, self.dk, self.dv = dmodel, dq, dk, dv
        self.heads = heads
        self.feedforward = feedforward

        self.Wq = nn.Linear(self.dmodel, self.heads*self.dq)
        self.Wk = nn.Linear(self.dmodel, self.heads*self.dk)
        self.Wv = nn.Linear(self.dmodel, self.heads*self.dv)
        self.unify = nn.Linear(self.heads*self.dv, self.dmodel)

        # Normalization
        self.norm1 = nn.LayerNorm(self.dmodel)

        # Feedforward
        self.ff = nn.Sequential(
                        nn.Linear(self.dmodel, self.feedforward),
                        nn.ReLU(),
                        nn.Linear(self.feedforward, self.dmodel)) 

        # Normalization
        self.norm2 = nn.LayerNorm(self.dmodel)


    def forward(self, x):
        """  
        @ params 
        - x => input torch tensor (d,n,dmodel)
        """

        attn = self.attention(x)
        norm1 = self.norm1(x + attn)
        feedfwd = self.ff(norm1)
        y = self.norm2(norm1 + feedfwd)

        return y


    def attention(self, x):
        """  
        @ params 
        - x => input torch tensor (d,n,dmodel)
        """

        queries = self.Wq(x)  # (d,n,h*dqk)
        keys = self.Wk(x)  # (d,n,h*dqk)
        values = self.Wv(x)  # (d,n,h*dv)

        scores = torch.bmm(queries, keys.transpose(1,2))/self.dk**0.5  # (d,n,n)

        attn = torch.bmm(scores, values)  # (d,n,h*dv)
        unified_attn = self.unify(attn)  # (d,n,dmodel)

        return unified_attn



class GENIACore(nn.Module):
    
    def __init__(self, dmodel, dq, dk, dv, heads, feedforward, vocab_size, num_categories=75, device="cuda"):
        super(GENIACore, self).__init__()

        self.dmodel, self.dq, self.dk, self.dv, self.heads, self.feedforward = dmodel, dq, dk, dv, heads, feedforward
        self.vocab_size = vocab_size
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.dmodel).to(self.device)
        self.transformer_enocder = TransformerEncoder(dmodel,dq, dk, dv,heads,feedforward).to(self.device)
        
        # Feedforward
        self.ff = nn.Sequential(
                        nn.Linear(self.dmodel, self.dmodel//2),
                        nn.ReLU(),
                        nn.Linear(self.dmodel//2, num_categories)).to(self.device) 

        # Normalization
        self.norm = nn.LayerNorm(num_categories).to(self.device)
    

    def forward(self, input_indices):
        """  
        @ params 
        - input_indices => input torch tensor (d,n)
        """
        UNK_IDX = 1
        d,n = input_indices.shape

        # Replace OOV word indices with the UNK index
        input_indices[input_indices>=self.vocab_size] = UNK_IDX
        
        # Convert the input indices into embeddings
        x = self.embedding(input_indices) + positionEncoding(d,n,self.dmodel, self.device)  # (d,n,dmodel)
        x = self.transformer_enocder(x)
        x = self.ff(x)
        x = self.norm(x)
        x = torch.sigmoid(x)

        return x



class GENIAModel(nn.Module):

    def __init__(self, dmodel, dq, dk, dv, heads, feedforward, vocab_size, num_categories=75, category_window_size=5, device="cuda") -> None:
        super(GENIAModel, self).__init__()

        assert(category_window_size%2 != 0), "category_window_size must be an odd number"
        self.num_categories = num_categories

        self.engine = GENIACore(dmodel, dq, dk, dv, heads, feedforward, vocab_size, num_categories, device )

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


    def forward(self, input_indices, target_labels):
        """  
        @params:
        - input_indices =>  pytorch tensor of dimension (d,n)
                            d sentences each containing n words
        - target_labels =>  one-hot encoded pytorch tensor of dimension (d,n,num_categories)
        """

        embeddings = self.engine.embedding(input_indices)
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


                        