import collections
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from transformers import logging
logging.set_verbosity_error()
from tqdm import tqdm
import os, sys, json
from utils import *
import matplotlib.pyplot as plt



class Logger:
    def __init__(self, log_dir) -> None:
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, "logs.json")

    
    def read_log(self):
        # return the contents of the log file
        try:
            with open(self.log_path, "r") as fp:
                file = json.load(fp)
            return file
        except:
            return dict()


    def write_log(self, new_logs:dict, replace:dict):

        # read the current log file
        log = self.read_log()

        for k,v in new_logs.items():

            # key already exists and value shouldn't be replaced
            if k in log and not replace[k]:
                log[k].append(v)

            # new key, but not replaceable => use list
            elif k not in log and not replace[k]:
                log[k] = [v]
            else:
                log[k] = v

        with open(self.log_path, "w") as fp:
            json.dump(log, fp)



class GENIAModel(nn.Module):

    def __init__(self, num_labels=11) -> None:
        super(GENIAModel, self).__init__()

        self.device = "cuda"
        self.model_dir = os.getcwd()
        self.model_name = "genia-model-v1.pt"
        self.lr = 0.01
        self.loss_amplify_factor = 10000
        self.lr_adaptive_factor = 0.5
        self.lr_patience = 5
        self.logger = Logger(self.model_dir)

        self.nl = num_labels
        self.label_dmodel = 64
        self.label_names = ['O','B-cell_type', 'I-cell_type', 'B-RNA', 'I-RNA', 'B-DNA', 'I-DNA', 
        'B-cell_line', 'I-cell_line', 'B-protein', 'I-protein']

        ## engine module layers

        # bert embeddings take care of the positional embeddings along with word embeddings
        self.bert_embeddings = BertModel.from_pretrained("bert-base-uncased").embeddings.to(self.device)

        self.num_engine_encoders = 6
        self.engine = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(d_model=768, nhead=4),
            ]*self.num_engine_encoders
        )

        self.engine_fc = nn.Sequential(
            nn.Linear(768, self.nl),
            nn.LayerNorm(self.nl)
        )
        
        ## global corrector module layers
        self.label_embeddings = nn.Embedding(self.nl, self.label_dmodel)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size= (self.nl,1)
        )

        self.gc = nn.TransformerEncoderLayer(d_model=self.label_dmodel + 768, nhead=4)
        self.gc_fc = nn.Sequential(
            nn.Linear(self.label_dmodel + 768, 768),
            nn.LayerNorm(768),
            nn.Linear(768, self.nl),
            nn.LayerNorm(self.nl)
        )


    def config(self, **params):
        for param, val in params.items():
            setattr(self, param, val) 


    def load(self):
        model_path = os.path.join(self.model_dir, self.model_name)
        self.load_state_dict(torch.load(model_path, map_location=self.device))


    def forward(self, x):
        """  
        @params:
        - x =>  tensor of dimension (d,n) containing word indices
        """

        d,n = x.shape

        # get the local-view probabilities for the ner task using the engine module
        word_embeddings = self.bert_embeddings(x)     # (d,n,768)
        state = self.engine(word_embeddings)          # (d,n,768)
        state = self.engine_fc(state)                 # (d,n,nl)
        probs = torch.sigmoid(state)                  # (d,n,nl)

        # generate weighted label embeddings by fusing label embeddings with the generated probabilities
        # 'probs' contain probabilities which represent the prediction confidence
        i = torch.LongTensor(range(self.nl)).repeat(d,n,1).to(self.device)  # (d,n,nl)
        label_embeddings = self.label_embeddings(i)                         # (d,n,nl, ldmodel)
        label_embeddings = torch.mul(
            probs.unsqueeze(-1), label_embeddings
        ).view(d*n, 1, self.nl, self.label_dmodel)                                # (d,n,nl,ldmodel) => (d*n, 1, nl, ldmodel) (n,c,h,w) 
        
        # convolve all prediction information into label_vec
        label_vec = self.conv1(label_embeddings).view(d,n,self.label_dmodel)       # (d*n, 1, nl, ldmodel) => (d*n,1,1,ldmodel) => (d,n,ldmodel) 
        
        # fuse the input word embeddings with prediction information in label_vec
        gc_vec = torch.concat((word_embeddings, label_vec), dim=-1) # (d,n,dmodel+768)

        # feedforward gc_vec through the global corrector module to get the final revised probabilities
        gc_last_hidden_state = self.gc(gc_vec)                     # (d,n,dmodel+768)
        gc_last_hidden_state = self.gc_fc(gc_last_hidden_state)    # (d,n,nl)

        # As x increases, sigmoid(x) tends to 1
        final_probs = torch.sigmoid(gc_last_hidden_state)
        
        return final_probs


    def calc_metrics(self, y_pred, y):
        """  
        @params:
        - y_pred    =>  probability tensor outputted by the model (d,n,nl)
        - y =>  tensor of 1's and 0's representing the actual labels (d,n,nl)
        """
        d,n,nl = y.shape
        N = d*n

        y = y.view(N, nl).float()
        y_pred = y_pred.view(N,nl).float()

        tp = torch.mul(y == y_pred, y_pred==1.0).sum(dim=0) # (nl,)
        tn = torch.mul(y == y_pred, y_pred==0.0).sum(dim=0) # (nl,)
        fp = torch.mul(y!=y_pred, y_pred==1.0).sum(dim=0)   # (nl,)
        fn = torch.mul(y!=y_pred, y_pred==0.0).sum(dim=0)   # (nl,)

        precision = tp/(tp+fp)  # (nl,)
        precision = precision.nan_to_num()
        precision[precision==torch.inf] = 0

        recall = tp/(tp+fn) # (nl,)                     
        recall = recall.nan_to_num()
        recall[recall==torch.inf] = 0

        f1_score = 2 / ((1/precision) + (1/recall)) # (nl,)

        num_positives = torch.count_nonzero(y, dim=0)
        num_negatives = N - num_positives

        metrics = collections.OrderedDict({
            "tp":tp.tolist(),
            "tn":tn.tolist(),
            "fp":fp.tolist(),
            "fn":fn.tolist(),
            "num-positives": num_positives.tolist(),
            "num-negatives": num_negatives.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1_score.tolist(),
            
        })

        # write the latest calculated metrics to a file
        with open(os.path.join(self.model_dir, "eval-results.txt"), "w") as fp:
            fp.write(pretty_print_results(metrics, self.label_names))

        with open(os.path.join(self.model_dir, "eval-results.csv"), "w") as fp:
            fp.write(get_csv(metrics, self.label_names))        

        return metrics


    def loss(self, y_pred, y):
        """  
        @params:
        - y_pred    =>  probability tensor outputted by the model (d,n,nl)
        - y =>  tensor of 1's and 0's representing the actual labels (d,n,nl)
        """
        d,n,nl = y.shape
        N = d*n

        eps = 1e-10

        y = y.view(N, nl)
        y_pred = y_pred.view(N,nl)

        # number of positive and negative examples for each class
        num_positives = torch.count_nonzero(y, dim=0)
        num_negatives = N - num_positives

        pos_weights = (num_negatives/(num_positives+eps)).view(1, nl).repeat(N,1) # (N,nl)
        neg_weights = (num_positives/(num_negatives+eps)).view(1, nl).repeat(N,1) # (N,nl)

        cost = torch.mul(
            torch.mul(pos_weights, y_pred),
            torch.mul(y, torch.log(y_pred + eps))
        ) + torch.mul(
            torch.mul(neg_weights, 1-y_pred),
            torch.mul(1-y, torch.log(1-y_pred + eps))
        )   # (N, nl)

        cost = -torch.sum(cost.view(-1), dim=0)
        cost = torch.divide(cost, N)*self.loss_amplify_factor
        return cost


    def test(self, test_loader):

        with torch.no_grad():

            y_pred = []
            y = []

            for _, (batch_x, batch_y) in enumerate(test_loader):

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                probs = self(batch_x)

                y_pred.append(probs)
                y.append(batch_y)

            y_pred = torch.concat(y_pred, dim=0)
            y = torch.concat(y, dim=0)

            thresholded_y_pred = y_pred.clone()
            thresholded_y_pred[thresholded_y_pred >= 0.5] = 1
            thresholded_y_pred[thresholded_y_pred < 0.5] = 0

            _ = self.calc_metrics(thresholded_y_pred, y)

            return self.loss(y_pred, y).item()


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
        
        progress_bar = tqdm(range(num_epochs), position=0, leave=True)

        avg_train_loss = 0.0    # loss over entire train set at each epoch
        avg_test_loss = 0.0     # loss over entire test set at each epoch
        num_batches = len(train_loader)

        for epoch in progress_bar:

            y_pred = y = []
            progress_bar.set_description(f"epoch: {epoch} ")

            # aggregate all batch losses in epoch_loss
            epoch_loss = 0.0

            for step, (batch_x, batch_y) in enumerate(train_loader):

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # clear gradients
                self.optimizer.zero_grad()

                # forward and backward pass
                probs = self(batch_x)

                single_batch_loss = self.loss(probs, batch_y)
                single_batch_loss.backward()
                
                # clip gradients and update weights
                torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
                self.optimizer.step()

                epoch_loss += single_batch_loss.detach().item()
                progress_bar.set_postfix(
                    {
                        "batch":step,
                        "batch-loss": str(round(single_batch_loss.detach().item(),4)),
                        "train-loss": str(round(avg_train_loss,4)),
                        "test-loss": str(round(avg_test_loss,4))
                    }
                )

                # collect all batch outputs for evaluation
                y_pred.append(probs)
                y.append(batch_y)
            
            y_pred = torch.concat(y_pred, dim=0)
            y = torch.concat(y, dim=0)
            
            d,n,nl = y_pred.shape
            N = d*n
            y = y.view(N, nl)
            y_pred = y_pred.view(N,nl)

            # adjust learning rate
            self.scheduler.step(epoch_loss)
            
            avg_test_loss = self.test(test_loader)
            avg_train_loss = epoch_loss/num_batches
            self.logger.write_log(
                {"train-losses": avg_train_loss, "test-losses": avg_test_loss}, 
                replace={"train-losses":False, "test-losses":False}
            )

            progress_bar.set_postfix(
                {
                    "batch":step,
                    "batch-loss": str(round(single_batch_loss.detach().item(),4)),
                    "train-loss": str(round(avg_train_loss,4)),
                    "test-loss": str(round(avg_test_loss,4))
                }
            )

            # save model after every epoch
            torch.save(self.state_dict(), os.path.join(self.model_dir, self.model_name))
        
        self.plot_train_test_loss_curve()
        self.plot_model_probs_histogram(y_pred)

        return avg_train_loss, avg_test_loss


    def plot_model_probs_histogram(self, y_pred):
        y_pred = y_pred.view(-1).cpu().tolist()
        plt.hist(y_pred, bins=20, range=(0,1))
        plot_path = os.path.join(self.model_dir, "model-probs-hist.png")
        if os.path.exists(plot_path):
            os.remove(plot_path)
        plt.savefig(plot_path)
        plt.close()

    
    def plot_train_test_loss_curve(self):
        log = self.logger.read_log()
        train_losses = log["train-losses"]
        test_losses = log["test-losses"]
        epochs = list(range(len(train_losses)))

        plt.plot(epochs, train_losses, "-b", label="Training loss")
        plt.plot(epochs, test_losses, "-r", label="Test loss")
        plt.legend(loc="upper right")

        plot_path = os.path.join(self.model_dir, "losses.png")
        if os.path.exists(plot_path):
            os.remove(plot_path)
        plt.savefig(plot_path)
        plt.close()



if __name__ == "__main__":
    m = GENIAModel()
    d,n,nc = 13,32,75
    x = torch.randint(0,20000,(d,n))
    print(m(x).shape)

    target = torch.randint(0,2,(d,n,nc))
    probs = torch.randn(d,n,nc).uniform_()

    l = m.loss(probs, target)
    print(l)