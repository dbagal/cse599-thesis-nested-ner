from reformer_pytorch import Reformer
import torch
import torch.nn as nn


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


class ReformerEncoder(nn.Module):
    
    def __init__(self, dmodel, dqk, dv, heads, feedforward, num_buckets):
        super(ReformerEncoder, self).__init__()

        self.dmodel, self.dqk, self.dv = dmodel, dqk, dv
        self.heads = heads
        self.feedforward = feedforward 
        self.num_buckets = num_buckets
        self.attn_in_place_penalty = 1e5

        assert(self.dmodel%2==0), "dmodel must be a multiple of 2!"
        assert(self.num_buckets%2==0), "num-buckets must be a multiple of 2!"

        self.Wqk = nn.Linear(self.dmodel//2, self.heads*self.dqk)
        self.Wv = nn.Linear(self.dmodel//2, self.heads*self.dv)
        self.unify = nn.Linear(self.heads*self.dv, self.dmodel//2)
        self.hashMatrix = torch.nn.Parameter(torch.randn(self.heads*self.dqk, self.num_buckets//2), requires_grad=False)

        # Normalization
        self.norm1 = nn.LayerNorm(self.dmodel//2)

        # Feedforward
        self.ff = nn.Sequential(
                        nn.Linear(self.dmodel//2, self.feedforward),
                        nn.ReLU(),
                        nn.Linear(self.feedforward, self.dmodel//2)) 

        # Normalization
        self.norm2 = nn.LayerNorm(self.dmodel//2)


    def forward(self, x):
        """  
        @ params 
        - x => input torch tensor (d,n,dmodel)
        """

        x1 = x[:,:,0:self.dmodel//2]  # (d,n,dmodel//2)
        x2 = x[:,:,self.dmodel//2:self.dmodel]  # (d,n,dmodel//2)
        
        attn_x2 = self.attention(x2)  # (d,n,dmodel//2)
        y1 = self.norm1(x1 + attn_x2)  # (d,n,dmodel//2)

        feedfwd = self.ff(y1)  # (d,n,dmodel//2)
        y2 = self.norm2(x2 + feedfwd)  # (d,n,dmodel//2)

        y = torch.cat((y1,y2), dim=-1)  # (d,n,dmodel)
        return y


    def attention(self, x):
        """  
        @ params 
        - x => input torch tensor (d,n,dmodel)
        """
        d = x.shape[0]
        n = x.shape[1]
        device = x.device
        
        assert(n%self.num_buckets==0), "Sequence length should be an integer multiple of num-buckets!"
        ch = 2*(n//self.num_buckets)  # Chunk Size
        nc = self.num_buckets//2  # No.of chunks

        qk = self.Wqk(x)  # (d,n,h*dqk)
        proj1 = torch.matmul(qk, self.hashMatrix)  # (d,n,b/2)
        proj2 = torch.matmul(-1*qk, self.hashMatrix)  # (d,n,b/2)
        hashes = torch.argmax(torch.cat((proj1,proj2),dim=-1), dim=-1)  # (d,n)
        sorted_indices = torch.sort(hashes, dim=-1).indices.view(-1)  # (d*n)

        offset = torch.arange(d).long()*n  # (d,)
        offset = offset.view(-1,1).repeat(1,n).view(-1)  # (d*n,)
        indices = offset.to(device) + sorted_indices  # (d*n,)
        
        # Sort qk according to the buckets
        qk = qk.view(-1, self.heads*self.dqk)[indices].view(d*nc,ch,self.heads*self.dqk)  # (d*nc,ch,h*dqk)

        scores = torch.bmm(qk, qk.transpose(1,2))/self.dqk**0.5  # (d*nc,ch,ch)
        diag = (1 + torch.eye(ch)*(self.attn_in_place_penalty - 1)).to(device) 
        scores = torch.true_divide(scores, diag)  # (d*nc,ch,ch)

        values = self.Wv(x)  # (d*nc,ch,h*dv)

        # Sort values according to buckets
        values = values.view(-1, self.heads*self.dv)[indices].view(d*nc,ch,self.heads*self.dv)  # (d*nc,ch,h*dqk)

        attn = torch.bmm(scores, values).view(d,n,self.heads*self.dv)  # (d,n,h*dv)
        unified_attn = self.unify(attn)  # (d,n,dmodel//2)

        return unified_attn


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


class GENIAReformer2(nn.Module):
    
    def __init__(self, dmodel, dqk, dv, heads, feedforward, vocab_size, num_buckets=32, num_bio_labels=77, device="cuda"):
        super(GENIAReformer2, self).__init__()

        self.dmodel, self.dqk, self.dv, self.heads, self.feedforward = dmodel, dqk, dv, heads, feedforward
        self.num_buckets = num_buckets
        self.vocab_size = vocab_size
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.dmodel).to(self.device)
        self.reformer_encoder = ReformerEncoder(dmodel,dqk,dv,heads,feedforward,num_buckets).to(self.device)
        
        # Feedforward
        self.ff = nn.Sequential(
                        nn.Linear(self.dmodel, self.dmodel//2),
                        nn.ReLU(),
                        nn.Linear(self.dmodel//2, num_bio_labels),
                        nn.ReLU(),).to(self.device) 

        # Normalization
        self.norm = nn.LayerNorm(num_bio_labels).to(self.device)
    

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
        x = self.reformer_encoder(x)
        x = self.ff(x)
        #x = self.norm(x)
        x = torch.sigmoid(x)

        return x



class GENIAReformer(nn.Module):
    
    def __init__(self, dmodel, depth, heads, vocab_size, num_buckets=32, num_bio_labels=77, device="cuda"):
        super(GENIAReformer, self).__init__()

        self.dmodel, self.heads = dmodel, heads
        self.num_buckets = num_buckets
        self.vocab_size = vocab_size
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.dmodel).to(self.device)
        self.reformer_encoder = Reformer(
            dim = dmodel,
            depth = depth,
            heads = heads,
            n_hashes=num_buckets
        ).to(self.device)

        # Feedforward
        self.ff = nn.Sequential(
                        nn.Linear(self.dmodel, self.dmodel//2),
                        nn.ReLU(),
                        nn.Linear(self.dmodel//2, num_bio_labels),
                        nn.ReLU(),).to(self.device) 

        # Normalization
        self.norm = nn.LayerNorm(num_bio_labels).to(self.device)
    

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
        x = self.reformer_encoder(x)
        x = self.ff(x)
        #x = self.norm(x)
        x = torch.sigmoid(x)

        return x


class GENIATransformer(nn.Module):
    
    def __init__(self, dmodel, dq, dk, dv, heads, feedforward, vocab_size, num_bio_labels=77, device="cuda"):
        super(GENIATransformer, self).__init__()

        self.dmodel, self.dq, self.dk, self.dv, self.heads, self.feedforward = dmodel, dq, dk, dv, heads, feedforward
        self.vocab_size = vocab_size
        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.dmodel).to(self.device)
        self.transformer_enocder = TransformerEncoder(dmodel,dq, dk, dv,heads,feedforward).to(self.device)
        
        # Feedforward
        self.ff = nn.Sequential(
                        nn.Linear(self.dmodel, self.dmodel//2),
                        nn.ReLU(),
                        nn.Linear(self.dmodel//2, num_bio_labels)).to(self.device) 

        # Normalization
        self.norm = nn.LayerNorm(num_bio_labels).to(self.device)
    

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
