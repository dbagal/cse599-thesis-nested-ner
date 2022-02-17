import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from transformers import BertModel
from transformers import logging
logging.set_verbosity_error()
from tqdm import tqdm
from metrics import Metrics


class GENIAModel(nn.Module):

    def __init__(self, dmodel=512, num_categories=75, ctxt_window_size=5, device="cpu") -> None:
        super(GENIAModel, self).__init__()

        assert(ctxt_window_size%2 != 0), "ctxt_window_size must be an odd number"

        self.device = device
        self.nc = num_categories
        self.dmodel = dmodel

        self.bert_engine = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.engine_fc = nn.Sequential(
            nn.Linear(768, num_categories),
            nn.LayerNorm(num_categories)
        )
         
        self.category_embeddings = nn.Embedding(num_categories, dmodel)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size= (self.nc,1)
        )

        self.cwnd = ctxt_window_size
        self.conv2 = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(self.cwnd, 1)
        )

        self.wagon = nn.TransformerEncoderLayer(d_model=self.dmodel + 768, nhead=4)

        self.wagon_fc = nn.Sequential(
            nn.Linear(dmodel+768, 768),
            nn.LayerNorm(768),
            nn.Linear(768, num_categories),
            nn.LayerNorm(num_categories)
        )

        self.metrics = Metrics([])


    def forward(self, input_ids):
        """  
        @params:
        - input_ids =>  indices of words in the input sequences (d,n)
        """
        d,n = input_ids.shape

        # get the regular probabilities for the ner task using the BERT engine
        state = self.bert_engine(input_ids = input_ids).last_hidden_state   # (d,n,768)
        state = self.engine_fc(state)                                       # (d,n,nc)
        probs = torch.sigmoid(state)                                        # (d,n,nc)

        # get all category embeddings to fuse them with the generated probabilities
        i = torch.LongTensor(range(self.nc)).repeat(d,n,1).to(self.device)  # (d,n,nc)
        category_embeddings = self.category_embeddings(i)                   # (d,n,nc, dmodel)
        
        # bring the category_embeddings in the form (n,c,h,w) for 2d convolution 
        # multiply the category embeddings with 'probs' which represents the prediction confidence for each of the category
        category_embeddings = torch.mul(
            probs.unsqueeze(-1), category_embeddings
        ).view(d*n, 1, self.nc, self.dmodel)    # (d*n, 1, nc, dmodel) => (d,n,nc, dmodel)

        # get the ctxt vector with every vector being a convolution of it's 'cwnd' surrounding vectors
        word_category_vec = self.conv1(category_embeddings).view(d,n,self.dmodel)       # (d*n, 1, nc, dmodel) => (d*n,1,1,dmodel) => (d,n,dmodel) 
        ctxt_vec = self.get_context_vector(word_category_vec, self.cwnd)                # (d,n,cwnd,dmodel)
        word_category_vec = self.conv2(
            ctxt_vec.view(d*n, 1, self.cwnd, self.dmodel)
        ).view(d,n,self.dmodel)                                                         # (d*n, 1, cwnd, dmodel) => (d*n,1,1,dmodel) => (d,n,dmodel)
        
        # get the original word embeddings from the BERT engine
        word_embeddings = self.bert_engine.embeddings(input_ids)    # (d,n,768)

        # form the wagon vector which is the fusion of the BERT embeddings and the weighted category embeddings 
        wagon_vec = torch.concat((word_embeddings, word_category_vec), dim=-1) # (d,n,dmodel+768)

        # feedforward the wagon vector through the wagon model to get the final revised probabilities
        wagon_last_hidden_state = self.wagon(wagon_vec)                     # (d,n,dmodel+768)
        wagon_last_hidden_state = self.wagon_fc(wagon_last_hidden_state)    # (d,n,num_categories)
        final_probs = torch.sigmoid(wagon_last_hidden_state)

        return final_probs


    def get_context_vector(self, vector, cwnd):
        """  
        @params:
        - vector    =>  pytorch tensor of dimension (d,n,dmodel)

        @returns:
        - ctxt_vec  =>  pytorch tensor of dimension (d,n,cwnd,dmodel)
        """

        d,n,dmodel = vector.shape

        # number of surrounding elements to the left or right
        # e.g: [0,0,23,0,0] => # of surrounding 0's = arr.size//2 = 5//2 = 2
        ns = cwnd//2

        # (d, n, dmodel) => (d, n + surround*2, dmodel)
        vector = torch.concat(
            [
                torch.zeros(d, ns, dmodel),
                vector,
                torch.zeros(d, ns, dmodel)
            ],
            dim =1
        )

        # calculate the number of blocks formed after grouping consecutive elements into groups of size cwnd
        # for 10 elements in the sequence, we can have 6 5-tuple sequences (10 - (5-1))
        # e.g: [1,2,3,4,5], cwnd = 3 => [ (1,2,3), (2,3,4), (3,4,5) ]
        # num_blocks = (n + ns*2)  - (self.cwnd - 1) = n

        # (d,n,dmodel) => (d, n, cwnd, dmodel)
        ctxt_vec = f.unfold(
            vector.view(d,1,n+2*ns,dmodel), 
            kernel_size=(cwnd, 1)
        ).view(d, n, cwnd, dmodel)

        return ctxt_vec


    def test(self, test_loader):
        for step, (input_ids, target_labels) in enumerate(test_loader):

            input_ids = input_ids.to(self.device)
            target_labels = target_labels.to(self.device)

            probs = self(input_ids)

            thresholds = self.metrics.get_optimal_threshold(target_labels, probs)
            metrics, metric_table = self.metrics.calc_metrics(target_labels, probs, thresholds)

 
    def train(self, train_loader, loss_fn,  num_epochs=10, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        progress_bar = tqdm(range(num_epochs), position=0, leave=True)

        recorded_e_loss = 0.0
        num_batches = len(train_loader)

        for epoch in progress_bar:
            progress_bar.set_description(f"epoch: {epoch} ")
            epoch_loss = 0.0

            for step, (input_ids, target_labels) in enumerate(train_loader):

                input_ids = input_ids.to(self.device)
                target_labels = target_labels.to(self.device)

                optimizer.zero_grad()

                probs = self(input_ids)

                batch_loss = loss_fn(probs, target_labels)
                batch_loss.backward()

                optimizer.step()

                epoch_loss += batch_loss.detach().item()

                progress_bar.set_postfix(
                    {
                        "batch":step,
                        "batch-loss": str(batch_loss.detach().item()),
                        "epoch-loss": str(recorded_e_loss)
                    }
                )

            recorded_e_loss = round(epoch_loss/num_batches, 8)
            progress_bar.set_postfix(
                {
                    "batch":step,
                    "batch-loss": str(batch_loss.detach().item()),
                    "epoch-loss": str(recorded_e_loss)
                }
            )



if __name__ == "__main__":
    m = GENIAModel()
    d,n = 13,32
    x = torch.randint(0,20000,(d,n))
    print(m(x).shape)