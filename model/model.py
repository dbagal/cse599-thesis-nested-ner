import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from transformers import BertModel
from transformers import logging
logging.set_verbosity_error()
from tqdm import tqdm
from metrics import Metrics
import os


class GENIAModel(nn.Module):

    def __init__(self, dmodel=64, ctxt_window_size=5, device="cpu") -> None:
        super(GENIAModel, self).__init__()

        assert(ctxt_window_size%2 != 0), "ctxt_window_size must be an odd number"

        self.device = device
        class_names = [
            'O', 'B_amino_acid_monomer', 'I_amino_acid_monomer', 'B_peptide', 'I_peptide', 'B_protein_N/A', 
            'I_protein_N/A', 'B_protein_complex', 'I_protein_complex', 'B_protein_domain_or_region', 
            'I_protein_domain_or_region', 'B_protein_family_or_group', 'I_protein_family_or_group', 
            'B_protein_molecule', 'I_protein_molecule', 'B_protein_substructure', 'I_protein_substructure', 
            'B_protein_subunit', 'I_protein_subunit', 'B_nucleotide', 'I_nucleotide', 'B_polynucleotide', 
            'I_polynucleotide', 'B_DNA_N/A', 'I_DNA_N/A', 'B_DNA_domain_or_region', 'I_DNA_domain_or_region', 
            'B_DNA_family_or_group', 'I_DNA_family_or_group', 'B_DNA_molecule', 'I_DNA_molecule', 
            'B_DNA_substructure', 'I_DNA_substructure', 'B_RNA_N/A', 'I_RNA_N/A', 'B_RNA_domain_or_region', 
            'I_RNA_domain_or_region', 'B_RNA_family_or_group', 'I_RNA_family_or_group', 'B_RNA_molecule', 
            'I_RNA_molecule', 'B_RNA_substructure', 'I_RNA_substructure', 'B_other_organic_compound', 
            'I_other_organic_compound', 'B_organic', 'I_organic', 'B_inorganic', 'I_inorganic', 'B_atom', 
            'I_atom', 'B_carbohydrate', 'I_carbohydrate', 'B_lipid', 'I_lipid', 'B_virus', 'I_virus', 
            'B_mono_cell', 'I_mono_cell', 'B_multi_cell', 'I_multi_cell', 'B_body_part', 'I_body_part', 
            'B_tissue', 'I_tissue', 'B_cell_type', 'I_cell_type', 'B_cell_component', 'I_cell_component', 
            'B_cell_line', 'I_cell_line', 'B_other_artificial_source', 'I_other_artificial_source', 
            'B_other_name', 'I_other_name']

        self.nc = len(class_names)
        self.dmodel = dmodel

        self.num_encoders = 3

        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.bert_model.embeddings.requires_grad = False

        self.engine = nn.Sequential(
            nn.Linear(768, 128),
            *[
                nn.TransformerEncoderLayer(d_model=128, nhead=4),
            ]*self.num_encoders
        )

        self.engine_fc = nn.Sequential(
            nn.Linear(128, self.nc),
            nn.LayerNorm(self.nc)
        )
         
        self.category_embeddings = nn.Embedding(self.nc, dmodel)

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
            nn.Linear(768, self.nc),
            nn.LayerNorm(self.nc)
        )

        self.metrics = Metrics(class_names)

        self.lr = 0.01
        self.model_name = "genia-model-v1.pt"


    def load(self, save_path):
        model_path = os.path.join(save_path, self.model_name)
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.eval()


    def forward(self, input_ids):
        """  
        @params:
        - input_ids =>  indices of words in the input sequences (d,n)
        """
        d,n = input_ids.shape

        # get the regular probabilities for the ner task using the BERT engine
        state = self.bert_model.embeddings(input_ids)    # (d,n,768)
        state = self.engine(state)
        #state = self.bert_model(input_ids = input_ids).last_hidden_state   # (d,n,768)
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
        word_embeddings = self.bert_model.embeddings(input_ids)    # (d,n,768)

        # form the wagon vector which is the fusion of the BERT embeddings and the weighted category embeddings 
        wagon_vec = torch.concat((word_embeddings, word_category_vec), dim=-1) # (d,n,dmodel+768)

        # feedforward the wagon vector through the wagon model to get the final revised probabilities
        wagon_last_hidden_state = self.wagon(wagon_vec)                     # (d,n,dmodel+768)
        wagon_last_hidden_state = self.wagon_fc(wagon_last_hidden_state)    # (d,n,nc)
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
    

    def loss(self, pred_probs, target_labels, p_factor = 0.5, n_factor = 2):
        """  
        @params:
        - pred_probs    =>  probability tensor outputted by the model (d,n,nc)
        - target_labels =>  tensor of 1's and 0's representing the actual labels (d,n,nc)
        - p_factor      =>  inversely proportional to the change in the loss for a positive label
        - n_factor      =>  inversely proportional to the change in the loss for a negative label
        """
        d,n,nc = pred_probs.shape
        N = d*n
        num_positives = [-1]*nc
        for i in range(nc):
            num_positives[i] = torch.count_nonzero(target_labels[:,:,i]==1).item()

        num_negatives = [N - num_positives[i] for i in range(nc)]

        p_weights = torch.FloatTensor([N/(p_factor * num_positives[i] + 1) for i in range(nc)]).unsqueeze(dim=-1) # (nc, 1)
        n_weights = torch.FloatTensor([N/(n_factor * num_negatives[i] + 1) for i in range(nc)]).unsqueeze(dim=-1) # (nc, 1)

        cost = torch.matmul(
            torch.mul( 
                target_labels, 
                torch.log(pred_probs) 
            ),
            p_weights
        ) + torch.matmul(
            torch.mul( 
                1 - target_labels, 
                torch.log(1 - pred_probs) 
            ),
            n_weights
        )

        cost = -torch.sum(cost.view(-1), dim=0)
        cost = torch.divide(cost, N)
        
        return cost


    def test(self, test_loader, thresholds):

        with torch.no_grad():
            predictions = []
            target_labels = []
        
            for _, (batch_input_ids, batch_target_labels) in enumerate(test_loader):

                batch_input_ids = batch_input_ids.to(self.device)
                batch_target_labels = batch_target_labels.to(self.device)

                probs = self(batch_input_ids)  # (d,n,nc)
                predictions.append(probs)
                target_labels.append(batch_target_labels)

            target_labels = torch.concat(target_labels, dim=0)
            predictions = torch.concat(predictions, dim=0)

            return self.metrics.calc_metrics(target_labels, predictions, thresholds)


 
    def train(self, train_loader, test_loader, save_path,  num_epochs=10):

        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        progress_bar = tqdm(range(num_epochs), position=0, leave=True)

        recorded_e_loss = 0.0
        
        num_batches = len(train_loader)

        for epoch in progress_bar:
            progress_bar.set_description(f"epoch: {epoch} ")
            epoch_loss = 0.0

            y_pred, y = [], []
            for step, (input_ids, target_labels) in enumerate(train_loader):

                input_ids = input_ids.to(self.device)
                target_labels = target_labels.to(self.device)

                optimizer.zero_grad()

                probs = self(input_ids)

                batch_loss = self.loss(probs, target_labels)
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

                y_pred.append(probs)
                y.append(target_labels)
            
            
            recorded_e_loss = epoch_loss/num_batches

            progress_bar.set_postfix(
                {
                    "batch":step,
                    "batch-loss": str(batch_loss.detach().item()),
                    "epoch-loss": str(recorded_e_loss)
                }
            )

            torch.save(self.state_dict(), os.path.join(save_path, self.model_name))

        y_pred = torch.concat(y_pred, dim=0)
        y = torch.concat(y, dim=0)

        thresholds = self.metrics.get_optimal_threshold(y_pred, y)

        _, metric_table = self.test(test_loader, thresholds)

        with open(os.path.join(save_path, "eval-results.txt"), "w") as fp:
            fp.write(metric_table)



if __name__ == "__main__":
    m = GENIAModel()
    d,n,nc = 13,32,75
    x = torch.randint(0,20000,(d,n))
    print(m(x).shape)

    target = torch.randint(0,2,(d,n,nc))
    probs = torch.randn(d,n,nc).uniform_()

    l = m.loss(probs, target)
    print(l)