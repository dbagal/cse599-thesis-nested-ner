import torch
from utils import *

class Metrics:

    def __init__(self, class_names, n_decimals=4) -> None:
        self.class_names = class_names
        self.n_decimals = n_decimals


    def calc_metrics(self, true_labels, predictions, thresholds):
        '''  
        @params:
        - true_labels: (d,n,num_categories) dimensional vector of 1s and 0s
        - predictions: (d,n,num_categories) dimensional probability vector
        '''
        
        num_categories = true_labels.shape[-1]

        true_positives, true_negatives, false_positives, false_negatives = [],[],[],[]
        num_examples, specificities, precisions, recalls, f1_scores = [], [], [], [], []

        for i in range(num_categories):
            predictions[predictions > thresholds[i]] = 1.0
            predictions[predictions <= thresholds[i]] = 0.0

            tp, tn, fp, fn = self.get_positives_and_negatives(true_labels[:,:,i], predictions[:,:,i])
            n = tp+tn+fp+fn
            precision = self.get_precision(tp, fp)
            recall = self.get_recall(tp, fn)
            specificity = self.get_specificity(fp, tn)
            f1_score = self.get_f1_score(precision, recall)

            true_positives.append(tp)
            true_negatives.append(tn)
            false_positives.append(fp)
            false_negatives.append(fn)
            num_examples.append(n)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1_scores.append(f1_score)

        headers = ['metric',] + self.class_names
        
        dataset = [
            ["true_positives"] + true_positives,
            ["true_negatives"] + true_negatives,
            ["false_positives"] + false_positives,
            ["false_negatives"] + false_negatives,
            ["num_examples"] + num_examples,
            ["precision"] + precisions,
            ["recall"] + recalls,
            ["f1_score"] + f1_scores
        ]

        metrics = {
            "true-positives": true_positives,
            "true-negatives":true_negatives,
            "false-positives":false_positives,
            "false-negatives":false_negatives,
            "num-examples":num_examples,
            "precision": precisions,
            "recall": recalls,
            "f1-score": f1_scores
        }
        
        return metrics, PrettyPrint.get_tabular_formatted_string(
                    dataset=dataset, 
                    headers=headers,
                    include_serial_numbers=False,
                    table_header="Evaluation metrics",
                    partitions=[5,7]
                )

    
    def get_optimal_threshold(self, true_labels, predictions, step=0.02):
        '''  
        @params:
        - true_labels: (d,n,num_categories) dimensional vector of 1s and 0s
        - predictions: (d,n,num_categories) dimensional probability vector
        '''
        thresholds = torch.range(0,1,step)
        num_categories = true_labels.shape[-1]

        max_youden_idx = [float("-inf"),]*num_categories
        final_thresh = [-1,]*num_categories

        for thresh in thresholds:
            predictions[predictions > thresh] = 1.0
            predictions[predictions <= thresh] = 0.0
               
            for i in range(num_categories):
                tp, tn, fp, fn = self.get_positives_and_negatives(true_labels[:,:,i], predictions[:,:,i])
                recall = self.get_recall(tp, fn)
                specificity = self.get_specificity(fp, tn)

                youden_idx = specificity + recall - 1
                if youden_idx > max_youden_idx[i]:
                    max_youden_idx[i] = youden_idx
                    final_thresh[i] = thresh

        return final_thresh, max_youden_idx
    

    def get_positives_and_negatives(self, true_labels, predictions):
        '''  
        @params:
        - true_labels: (d,n,) dimensional vector of 1s and 0s
        - predictions: (d,n,) dimensional vector of 1s and 0s

        @function:
        - get true positives, true negatives, false positives, false negatives for ith class
        '''
        target = true_labels.view(-1)
        preds = predictions.view(-1)
        tp = torch.mul(target == preds, preds==1.0).sum().item()
        tn = torch.mul(target == preds, preds==0.0).sum().item()
        fp = torch.mul(target==0.0, preds==1.0).sum().item()
        fn = torch.mul(target==1.0, preds==0.0).sum().item()

        return tp, tn, fp, fn


    def get_precision(self, tp, fp):
        if tp!=0:
            return round(tp/(tp+fp), self.n_decimals)
        else:
            return 0

    
    def get_recall(self, tp, fn):
        if tp!=0:
            return round(tp/(tp+fn), self.n_decimals)
        else:
            return 0


    def get_specificity(self, fp, tn):
        return round(tn/(tn+fp), self.n_decimals)


    def get_accuracy(self, tp, tn, fp, fn):
        return round((tp+tn)/(tp+tn+fp+fn), self.n_decimals)


    def get_f1_score(self,precision, recall):
        if precision == 0 or recall == 0:
            return None
        else:
            return round(2 / ((1/precision) + (1/recall)), self.n_decimals)

    