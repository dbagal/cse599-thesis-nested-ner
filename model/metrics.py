# pytorch-ignite == 0.4.8

import collections
import torch
from utils import *


class Metrics:

    def __init__(self, class_names, n_decimals=4) -> None:
        self.class_names = class_names
        self.n_decimals = n_decimals


    def calc_metrics(self, y, y_pred, thresholds):
        '''  
        @params:
        - y: (d,n,num_categories) dimensional vector of 1s and 0s
        - y_pred: (d,n,num_categories) dimensional probability vector
        '''
        y_pred[y_pred > thresholds] = 1
        y_pred[y_pred <= thresholds] = 0

        tp, fp, tn, fn = self.get_cm_elements(y_pred, y) # (nc,)

        num_examples = tp+tn+fp+fn

        precision = tp/(tp+fp)
        precision = precision.nan_to_num()
        precision[precision==torch.inf] = 0

        recall = tp/(tp+fn)
        recall = recall.nan_to_num()
        recall[recall==torch.inf] = 0

        specificity = tn/(tn+fp)
        f1_score = 2 / ((1/precision) + (1/recall))

        round_tensor = lambda tensor: [round(x, self.n_decimals) for x in tensor.tolist()]

        metrics = collections.OrderedDict({
            "true-positives": tp.tolist(),
            "true-negatives":tn.tolist(),
            "false-positives":fp.tolist(),
            "false-negatives":fn.tolist(),
            "num-examples":num_examples.tolist(),
            "specificity": round_tensor(specificity),
            "precision": round_tensor(precision),
            "recall": round_tensor(recall),
            "f1-score": round_tensor(f1_score)
        })

        headers = ['metric',] + self.class_names
        dataset = [[key]+val for key, val in metrics.items()]

        return metrics, PrettyPrint.get_tabular_formatted_string(
                    dataset=dataset, 
                    headers=headers,
                    include_serial_numbers=False,
                    table_header="Evaluation metrics",
                    partitions=[5,7]
                )


    def get_optimal_threshold(self, y, y_pred, step=0.1):
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
        
        return tp, fp, tn, fn



        
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

    