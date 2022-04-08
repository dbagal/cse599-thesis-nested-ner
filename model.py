from pickle import dump, load
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from transformers import logging
logging.set_verbosity_error()
from tqdm import tqdm
from metrics import Metrics
import os, sys
import matplotlib.pyplot as plt


class GENIAModel(nn.Module):

    def __init__(self, save_path, dmodel=64, device="cuda") -> None:
        super(GENIAModel, self).__init__()

        self.device = device
        self.save_path = save_path

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
        self.metrics = Metrics(class_names, save_path)

        self.nc = len(class_names)
        self.dmodel = dmodel

        # model layers
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.bert_embeddings = self.bert_model.embeddings

        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)

        self.num_encoders = 6
        self.engine = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(d_model=768, nhead=4),
            ]*self.num_encoders
        )

        self.engine_fc = nn.Sequential(
            nn.Linear(768, self.nc),
            nn.LayerNorm(self.nc)
        )
         
        self.category_embeddings = nn.Embedding(self.nc, dmodel)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size= (self.nc,1)
        )

        self.wagon = nn.TransformerEncoderLayer(d_model=self.dmodel + 768, nhead=4)
        """ self.wagon = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(d_model=self.dmodel + 768, nhead=4),
            ]*self.num_encoders
        ) """

        self.wagon_fc = nn.Sequential(
            nn.Linear(dmodel+768, 768),
            nn.LayerNorm(768),
            nn.Linear(768, self.nc),
            nn.LayerNorm(self.nc)
        )
        
        # model parameters
        self.lr = 0.01
        self.p_factor = 0.5
        self.n_factor = 1
        self.loss_amplify_factor = 10000
        self.lr_adaptive_factor = 0.5
        self.lr_patience = 5
        self.model_name = "genia-model-v1.pt"


    def load(self):
        model_path = os.path.join(self.save_path, self.model_name)
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        #self.eval()


    def forward(self, input_ids):
        """  
        @params:
        - input_ids =>  indices of words in the input sequences (d,n)
        """
        d,n = input_ids.shape

        # get the regular probabilities for the ner task using the BERT engine
        state = self.bert_embeddings(input_ids)     # (d,n,768)
        #state = self.norm1(state)                   # (d,n,768)
        state = self.engine(state)                  # (d,n,768)

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

        word_category_vec = self.conv1(category_embeddings).view(d,n,self.dmodel)       # (d*n, 1, nc, dmodel) => (d*n,1,1,dmodel) => (d,n,dmodel) 
        
        # get the original word embeddings from the BERT engine
        word_embeddings = self.bert_embeddings(input_ids)    # (d,n,768)
        #word_embeddings = self.norm2(word_embeddings)

        # form the wagon vector which is the fusion of the BERT embeddings and the weighted category embeddings 
        wagon_vec = torch.concat((word_embeddings, word_category_vec), dim=-1) # (d,n,dmodel+768)

        # feedforward the wagon vector through the wagon model to get the final revised probabilities
        wagon_last_hidden_state = self.wagon(wagon_vec)                     # (d,n,dmodel+768)
        wagon_last_hidden_state = self.wagon_fc(wagon_last_hidden_state)    # (d,n,nc)

        # x -> inf => sigmoid(x) -> 1
        final_probs = torch.sigmoid(wagon_last_hidden_state)
        
        return final_probs


    def loss(self, y_hat, y):
        """  
        @params:
        - y_hat    =>  probability tensor outputted by the model (d,n,nc)
        - y =>  tensor of 1's and 0's representing the actual labels (d,n,nc)
        """
        d,n,nc = y_hat.shape
        N = d*n

        epsilon = 1e-10

        y = y.view(N, nc)
        y_hat = y_hat.view(N,nc)

        # store the number of positive and negative examples for each class in num_positives and num_negatives
        num_positives = torch.count_nonzero(y, dim=0)
        num_negatives = N - num_positives

        #pos_weights = (num_negatives*100/N).view(1, nc).repeat(N,1) # (N,nc)
        #neg_weights = (num_positives*100/N).view(1, nc).repeat(N,1) # (N,nc)

        #class_weights = torch.log(epsilon + N/(num_positives + epsilon)) + 1  # (nc,)
        #class_weights = 1 - num_positives/N  # (nc,)
        #class_weights = class_weights.view(1, nc).repeat(N,1)   # (N,nc)

        pos_weights = (num_negatives/(num_positives+epsilon)).view(1, nc).repeat(N,1) # (N,nc)
        neg_weights = (num_positives/(num_negatives+epsilon)).view(1, nc).repeat(N,1) # (N,nc)

        pos_weights = torch.mul(pos_weights, y_hat)
        neg_weights = torch.mul(neg_weights, 1-y_hat)

        cost = torch.mul(
            pos_weights,
            torch.mul(y, torch.log(y_hat + epsilon))
        ) + torch.mul(
            neg_weights,
            torch.mul(1-y, torch.log(1-y_hat + epsilon))
        )   # (N, nc)

        #cost = torch.mul(class_weights, cost) # (N,nc)

        cost = -torch.sum(cost.view(-1), dim=0)
        cost = torch.divide(cost, N)*self.loss_amplify_factor
        return cost


    def dynamic_weighted_bce(self, pred_probs, target_labels):
        """  
        @params:
        - pred_probs    =>  probability tensor outputted by the model (d,n,nc)
        - target_labels =>  tensor of 1's and 0's representing the actual labels (d,n,nc)
        """
        d,n,nc = pred_probs.shape
        N = d*n

        epsilon = 1e-10

        target_labels = target_labels.view(N, nc)
        pred_probs = pred_probs.view(N,nc)

        # store the number of positive and negative examples for each class in num_positives and num_negatives
        num_positives = torch.count_nonzero(target_labels, dim=0)
        num_negatives = N - num_positives

        weights_pos = torch.log(epsilon + torch.max(num_positives)/(num_positives + epsilon)) + 1  # (nc,)
        weights_pos = weights_pos.view(1, nc).repeat(N,1)   # (N,nc)

        weights_neg = torch.log(epsilon + torch.max(num_negatives)/(num_negatives + epsilon)) + 1  # (nc,)
        weights_neg = weights_neg.view(1, nc).repeat(N,1)   # (N,nc)

        dynamic_weights_pos = torch.pow(weights_pos, 1-pred_probs)
        dynamic_weights_neg = torch.pow(weights_neg, pred_probs)

        cost = torch.mul(
            dynamic_weights_pos,
            torch.mul(target_labels, torch.log(pred_probs + epsilon))
        ) + torch.mul(
            dynamic_weights_neg,
            torch.mul(1-target_labels, torch.log(1-pred_probs + epsilon))
        )  # (N, nc)

        cost = -torch.sum(cost.view(-1), dim=0)
        cost = torch.divide(cost, N)*self.loss_amplify_factor
        return cost
        

    def weighted_bce(self, pred_probs, target_labels):
        """  
        @params:
        - pred_probs    =>  probability tensor outputted by the model (d,n,nc)
        - target_labels =>  tensor of 1's and 0's representing the actual labels (d,n,nc)
        - p_factor      =>  inversely proportional to the change in the loss for a positive label
        - n_factor      =>  inversely proportional to the change in the loss for a negative label
        """
        d,n,nc = pred_probs.shape
        N = d*n

        # store the number of positive and negative examples for each class in num_positives and num_negatives
        num_positives = [-1]*nc
        for i in range(nc):
            num_positives[i] = torch.count_nonzero(target_labels[:,:,i]==1).item()

        num_negatives = [N - num_positives[i] for i in range(nc)]

        # each class has a p_weight and n_weight
        # p_weight amount of ylogy and n_weight amount of (1-y)log(1-y) is added to the loss
        p_weights = torch.FloatTensor([N/(self.p_factor * num_positives[i] + 0.001) for i in range(nc)]).unsqueeze(dim=-1).to(self.device) # (nc, 1)
        n_weights = torch.FloatTensor([N/(self.n_factor * num_negatives[i] + 0.001) for i in range(nc)]).unsqueeze(dim=-1).to(self.device) # (nc, 1)

        # calculate bce loss as p_weight * ylogy + n_weight * (1-y)log(1-y)
        # remember, if pred_probs is 1 => log(1-pred_probs) will be -inf which will further give cost as Nan values
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
        cost = torch.divide(cost, N)*self.loss_amplify_factor

        return cost

    
    def test(self, test_loader, thresholds):

        with torch.no_grad():

            predictions = []
            target_labels = []
        
            for _, (batch_input_ids, batch_target_labels) in enumerate(test_loader):

                batch_input_ids = batch_input_ids.to(self.device)
                batch_target_labels = batch_target_labels.to(self.device)

                # calc output probabilities for each batch input
                probs = self(batch_input_ids)  # (d,n,nc)

                # store the batch probabilities and batch target labels for calculating metrics over the entire test set
                predictions.append(probs)
                target_labels.append(batch_target_labels)

            # concatenate all batch outputs
            target_labels = torch.concat(target_labels, dim=0)
            predictions = torch.concat(predictions, dim=0)

            # calculate metrics for the entire test set
            self.metrics.calc_metrics(target_labels, predictions, thresholds)

            return self.loss(predictions, target_labels).item()

 
    def train(self, train_loader, test_loader,  num_epochs=10):

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=self.lr_adaptive_factor, 
            patience=self.lr_patience, 
            threshold=0.0001, 
            threshold_mode='abs'
        )
        #self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1, cycle_momentum=False)
        progress_bar = tqdm(range(num_epochs), position=0, leave=True)

        # train_loss and test_loss represents the loss over the entire training set and test set after each epoch
        train_loss = 0.0
        test_loss = 0.0
        
        num_batches = len(train_loader)

        for epoch in progress_bar:

            y_pred = y = []
            progress_bar.set_description(f"epoch: {epoch} ")

            # aggregate all batch losses in epoch_loss
            epoch_loss = 0.0

            for step, (input_ids, target_labels) in enumerate(train_loader):
                
                input_ids = input_ids.to(self.device)
                target_labels = target_labels.to(self.device)

                # clear gradients
                self.optimizer.zero_grad()

                # forward and backward pass
                probs = self(input_ids)

                batch_loss = self.loss(probs, target_labels)
                batch_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
                self.optimizer.step()

                epoch_loss += batch_loss.detach().item()
                progress_bar.set_postfix(
                    {
                        "batch":step,
                        "batch-loss": str(round(batch_loss.detach().item(),4)),
                        "train-loss": str(round(train_loss,4)),
                        "test-loss": str(round(test_loss,4))
                    }
                )

                # collect all batch outputs for evaluation
                y_pred.append(probs)
                y.append(target_labels)
            
            # calculate optimal threshold for each class based on the predictions and true labels of entire training set
            y_pred = torch.concat(y_pred, dim=0).to(self.device)
            y = torch.concat(y, dim=0).to(self.device)

            d,n,nc = y_pred.shape
            N = d*n
            y = y.view(N, nc)
            y_pred = y_pred.view(N,nc)

            # store the number of positive and negative examples for each class in num_positives and num_negatives
            #num_positives = torch.count_nonzero(y, dim=0)   # (nc, )
            #num_negatives = N - num_positives               # (nc, )

            #thresholds = num_positives/N
            #thresholds = torch.minimum(num_positives, num_negatives)/torch.maximum(num_positives, num_negatives)
            thresholds = self.metrics.get_optimal_threshold(y_pred, y).to(self.device)

            self.scheduler.step(epoch_loss)
            
            # calculate test loss and all metrics
            test_loss = self.test(test_loader, thresholds)
            train_loss = epoch_loss/num_batches

            self.metrics.record("train-loss", train_loss)
            self.metrics.record("test-loss", test_loss)

            progress_bar.set_postfix(
                {
                    "batch":step,
                    "batch-loss": str(round(batch_loss.detach().item(),4)),
                    "train-loss": str(round(train_loss,4)),
                    "test-loss": str(round(test_loss,4))
                }
            )

            # save model after every epoch
            torch.save(self.state_dict(), os.path.join(self.save_path, self.model_name))
        
        self.metrics.write_metrics()

        # write the latest calculated metrics to a file
        with open(os.path.join(self.save_path, "eval-results.txt"), "w") as fp:
            fp.write(self.metrics.pretty_print_results())

        with open(os.path.join(self.save_path, "eval-results.csv"), "w") as fp:
            fp.write(self.metrics.get_csv())

        # save the model probabilities as a pickle file
        with open(os.path.join(self.save_path, "model-probs.tensor"), "wb") as fp:
            dump(y_pred, fp)

        # save the dataset target labels as a pickle file
        with open(os.path.join(self.save_path, "dataset-labels.tensor"), "wb") as fp:
            dump(y, fp)

        train_losses = self.metrics.metric_file["train-loss"]
        test_losses = self.metrics.metric_file["test-loss"]
        epochs = list(range(len(train_losses)))

        plt.plot(epochs, train_losses, "-b", label="Training loss")
        plt.plot(epochs, test_losses, "-r", label="Test loss")
        plt.legend(loc="upper right")
        plt.savefig("losses.png")

        return train_loss, test_loss


if __name__ == "__main__":
    m = GENIAModel()
    d,n,nc = 13,32,75
    x = torch.randint(0,20000,(d,n))
    print(m(x).shape)

    target = torch.randint(0,2,(d,n,nc))
    probs = torch.randn(d,n,nc).uniform_()

    l = m.loss(probs, target)
    print(l)