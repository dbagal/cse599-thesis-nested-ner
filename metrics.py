# pytorch-ignite == 0.4.8

import collections
import os
import torch
import json
from utils import *


class Metrics:

    def __init__(self, class_names, save_folder, n_decimals=4, fname="metrics", device="cuda") -> None:
        """  
        @params:
        - class_names   =>  list of names of all classes
        - save_folder   =>  path to the folder where metrics file is stored / should be stored
        - n_decimals    =>  number of decimals needed in the metrics
        - fname         =>  name of the file where all metrics will be stored
        """
        self.class_names = class_names
        self.n_decimals = n_decimals
        self.device = device
        
        self.save_folder = save_folder
        self.metric_file_name = os.path.join(self.save_folder, fname+".json")

        # store the json data in self.metric_file
        self.read_metrics()

        # store the calculated metrics
        self.calculated_metrics = None
    

    def read_metrics(self):
        try:
            with open(self.metric_file_name, "r") as fp:
                file = json.load(fp)
            self.metric_file = file
        except:
            self.metric_file = None

    
    def write_metrics(self):
        with open(self.metric_file_name, "w") as fp:
            json.dump(self.metric_file, fp)


    def record(self, metric_name, val):
        """  
        @description:
        - records values 'val' for 'metric_name' in the metric file
        """
        if not self.metric_file:
            self.metric_file = {
                metric_name: [val]
            }

        elif metric_name not in self.metric_file:
            self.metric_file[metric_name] = [val]
        
        elif metric_name in self.metric_file:
            self.metric_file[metric_name].append(val)

        self.calculated_metrics[metric_name] = val
    

    def calc_metrics(self, y, pred_probs, thresholds):
        '''  
        @params:
        - y: (d,n,num_categories) dimensional vector of 1s and 0s
        - y_pred: (d,n,num_categories) dimensional probability vector
        '''

        y_pred = pred_probs.clone()

        y_pred[y_pred > thresholds] = 1
        y_pred[y_pred <= thresholds] = 0

        tp, fp, tn, fn = self.get_cm_elements(y_pred, y) # (nc,)

        _,_,nc = y.shape
        num_examples = [-1]*nc
        for i in range(nc):
            num_examples[i] = torch.count_nonzero(y[:,:,i]==1).item()

        precision = tp/(tp+fp)
        precision = precision.nan_to_num()
        precision[precision==torch.inf] = 0

        recall = tp/(tp+fn)
        recall = recall.nan_to_num()
        recall[recall==torch.inf] = 0

        specificity = tn/(tn+fp)
        f1_score = 2 / ((1/precision) + (1/recall))

        round_tensor = lambda tensor: [round(x, self.n_decimals) for x in tensor.tolist()]

        self.calculated_metrics = collections.OrderedDict({
            "true-positives": tp.tolist(),
            "true-negatives":tn.tolist(),
            "false-positives":fp.tolist(),
            "false-negatives":fn.tolist(),
            "num-examples":num_examples,
            "specificity": round_tensor(specificity),
            "precision": round_tensor(precision),
            "recall": round_tensor(recall),
            "f1-score": round_tensor(f1_score)
        })


    def get_csv(self):
        file = f"metric,{','.join(self.class_names)}"
        for metric, vals in self.calculated_metrics.items():
            file += f"\n{metric},"
            if type(vals)==list:
                file += f"{','.join([str(val) for val in vals])}"
            else:
                file += f"{vals}"
        return file
        
            
    def pretty_print_results(self):
        if self.calculated_metrics:
            headers = ['metric',] + self.class_names
            
            dataset = []
            for key, val in self.calculated_metrics.items():
                if type(val) == list and len(val)==len(self.class_names):
                    dataset += [[key]+val]
                else:
                    dataset += [[key]+[val]+[0,]*(len(self.class_names)-1)]

            file = PrettyPrint.get_tabular_formatted_string(
                        dataset=dataset, 
                        headers=headers,
                        include_serial_numbers=False,
                        table_header="Evaluation metrics",
                        partitions=[5,7]
                    )
            return file
        return None


    def get_optimal_threshold(self, y, y_pred, num_samples=50, sample_proportion=0.4, start=0.1, end=0.9, step=0.05):
        '''  
        @params:
        - y: (d,n,num_categories) dimensional vector of 1s and 0s
        - y_pred: (d,n,num_categories) dimensional probability vector
        '''
        thresholds = torch.arange(start,end,step)
        sample_size = int(y.shape[0] * sample_proportion)

        thresh_cohens_kappas = []

        for thresh in thresholds:
            y_hat = y_pred

            cohens_kappas = []
            for _ in range(num_samples):
                indices = torch.randint(0, y.shape[0], (sample_size,)).to(self.device)
                y_hat = y_pred[indices, :]  # (sample_size, nc)
                y_hat[y_hat > thresh] = 1
                y_hat[y_hat <= thresh] = 0

                tp, fp, tn, fn = self.get_cm_elements(y_pred, y) 

                n = tp+fp+tn+fn

                rel_agreement_prob = (tp + tn)/n
                chance_agreement_prob = (((tp + fp) * (tp + fn)) + ((tn + fp) * (tn + fn)))/ n*n

                cohens_kappa = (rel_agreement_prob - chance_agreement_prob)/(1 - chance_agreement_prob)
            
                cohens_kappas.append(cohens_kappa)
            
            cohens_kappas = torch.concat(cohens_kappas, dim=0).view(num_samples, -1).to(self.device)
            thresh_cohens_kappas.append(torch.median(cohens_kappas, dim=0).values)

        thresh_cohens_kappas = torch.concat(thresh_cohens_kappas, dim=0).view(thresholds.shape[0], -1).to(self.device)  # (num_threshs, nc)
        thresh_indices = torch.max(thresh_cohens_kappas, dim=0).indices

        return thresholds[thresh_indices]
        

    def get_youden_threshold(self, y, y_pred, step=0.1):
        '''  
        @params:
        - y: (d,n,num_categories) dimensional vector of 1s and 0s
        - y_pred: (d,n,num_categories) dimensional probability vector
        '''
        thresholds = torch.range(0,1,step)

        youden_indices = []

        for thresh in thresholds:
            y_pred[y_pred > thresh] = 1
            y_pred[y_pred <= thresh] = 0

            tp, fp, tn, fn = self.get_cm_elements(y_pred, y)
            recall = tp/(tp+fn)
            recall = recall.nan_to_num()
            recall[recall==torch.inf] = 0

            specificity = tn/(tn+fp)

            youden_idx = specificity + recall - 1
            youden_indices.append(youden_idx.view(1,-1))

        youden_indices = torch.concat(youden_indices, dim=0)
        max_youden_idx = torch.max(youden_indices, dim=0).indices

        return thresholds[max_youden_idx]


    def get_cm_elements(self, y_pred, y):
        '''  
        @params:
        - y: (d,n,num_categories) dimensional vector of 1s and 0s
        - y_pred: (d,n,num_categories) dimensional probability vector
        '''
        nc = y_pred.shape[-1]

        y_pred = y_pred.view(-1,nc)
        y = y.view(-1, nc)

        reshaped_true_labels = y.transpose(0,1).reshape(nc, -1)
        reshaped_pred = y_pred.transpose(0,1).reshape(nc,-1)

        true_labels_total = reshaped_true_labels.sum(dim=1)
        pred_total = reshaped_pred.sum(dim=1)

        tp = (reshaped_true_labels * reshaped_pred).sum(dim=1)
        fp = pred_total - tp
        fn = true_labels_total - tp
        tn = reshaped_true_labels.shape[1] - tp - fp - fn
        
        return tp.int(), fp.int(), tn.int(), fn.int()


        
if __name__ == "__main__":
    m = Metrics(['a','b','c','d','e','f','g','h'])
    d,n,nc = 2,3,8

    pred = torch.randn(d,n,nc).uniform_()
    #pred = torch.randint(0,2,(d,n,nc))
    y = torch.randint(0,2,(d,n,nc))
    t = torch.randn(8).uniform_()
    _, s= m.calc_metrics(pred, y, t)
    print(pred)
    print(t)
    pred[pred > t] = 1
    pred[pred <= t] = 0
    print(pred)
    #print(s)

    