import torch
from utils import *

class Metrics:

    def __init__(self, n_decimals=4) -> None:
        '''  
        @params:
        - labels: (d,n,num_categories) dimensional vector of 1s and 0s
        - predictions: (d,n,num_categories) dimensional vector of 1s and 0s
        '''
        self.n_decimals = n_decimals
    

    def set_metrics(self, labels, predictions):
        '''  
        @params:
        - labels: (d,n,num_categories) dimensional vector of 1s and 0s
        - predictions: (d,n,num_categories) dimensional vector of 1s and 0s
        '''
        labels = labels.contiguous()
        predictions = predictions.contiguous()

        self.precision = []
        self.recall = []
        self.accuracy = []
        self.f1_score = []
        self.true_positives = []
        self.true_negatives = []
        self.false_positives = []
        self.false_negatives = []

        num_categories = labels.shape[-1]

        for i in range(num_categories):
            target = labels[:,:,i].view(-1)
            preds = predictions[:,:,i].view(-1)
            tp = torch.mul(target == preds, preds==1 ).sum().item()
            tn = torch.mul(target == preds, preds==0 ).sum().item()
            fp = torch.mul(target==0, preds==1).sum().item()
            fn = torch.mul(target==0, preds==1).sum().item()

            self.true_positives += [tp]
            self.true_negatives += [tn]
            self.false_positives += [fp]
            self.false_negatives += [fn]

            precision = self.get_precision(tp, fp)
            recall = self.get_recall(tp,fn)
            acc = self.get_accuracy(tp, tn, fp, fn)
            f1_score = self.get_f1_score(precision, recall)
            self.precision += [precision]
            self.recall += [recall]
            self.accuracy += [acc]
            self.f1_score += [f1_score] 
            
    
    def update_metrics(self, labels, predictions):
        '''  
        @params:
        - labels: (d,n,num_categories) dimensional vector of 1s and 0s
        - predictions: (d,n,num_categories) dimensional vector of 1s and 0s
        '''
        labels = labels.contiguous()
        predictions = predictions.contiguous()

        assert(labels.shape[-1]==len(self.accuracy)), "Inconsistency in number of classes"

        num_categories = labels.shape[-1]
        
        for i in range(num_categories):
            target = labels[:,:,i].view(-1)
            preds = predictions[:,:,i].view(-1)
            tp = torch.mul(target == preds, preds==1).sum().item()
            tn = torch.mul(target == preds, preds==0).sum().item()
            fp = torch.mul(target==0, preds==1).sum().item()
            fn = torch.mul(target==1, preds==0).sum().item()

            self.true_positives[i] += tp
            self.true_negatives[i] += tn
            self.false_positives[i] += fp
            self.false_negatives[i] += fn

        self.precision = []
        self.recall = []
        self.accuracy = []
        self.f1_score = []

        for tp,tn,fp,fn in zip(self.true_positives, self.true_negatives, self.false_positives, self.false_negatives): 
            precision = self.get_precision(tp, fp)
            recall = self.get_recall(tp,fn)
            acc = self.get_accuracy(tp, tn, fp, fn)
            f1_score = self.get_f1_score(precision, recall)

            self.precision += [precision]
            self.recall += [recall]
            self.accuracy += [acc]
            self.f1_score += [f1_score] 


    def get_precision(self, tp, fp):
        return round(tp/(tp+fp), self.n_decimals)

    
    def get_recall(self, tp, fn):
        return round(tp/(tp+fn), self.n_decimals)


    def get_accuracy(self, tp, tn, fp, fn):
        return round((tp+tn)/(tp+tn+fp+fn), self.n_decimals)


    def get_f1_score(self,precision, recall):
        if precision == 0 and recall == 0:
            return None
        else:
            return round(2 / ((1/precision) + (1/recall)), self.n_decimals)

    
    def __str__(self) -> str:
        
        num_categories = len(self.accuracy)

        headers = ['metric',]
        for i in range(num_categories):
            headers += [f"class_{i}"]
        
        dataset = [
            ["tp"] + self.true_positives,
            ["tn"] + self.true_negatives,
            ["fp"] + self.false_positives,
            ["fn"] + self.false_negatives,
            ["acc"] + self.accuracy,
            ["precision"] + self.precision,
            ["recall"] + self.recall,
            ["f1_score"] + self.f1_score
        ]
        
        return PrettyPrint.get_tabular_formatted_string(
                    dataset=dataset, 
                    headers=headers,
                    include_serial_numbers=False,
                    table_header="Evaluation metrics"
                ) 


if __name__=="__main__":
    labels = torch.tensor(
        [
            # sentence 1
            [
                [1,0,0,1,0], # word 1
                [1,1,0,1,0], # word 2
                [0,0,0,1,0]  # word 3
            ],
            # sentence 2
            [
                [0,0,0,1,1],
                [1,0,1,1,0],
                [0,1,0,1,0]
            ]
        ] 
    )

    preds = torch.tensor(
        [
            # sentence 1
            [
                [1,0,1,0,0], # word 1
                [1,0,0,1,0], # word 2
                [0,0,0,1,1]  # word 3
            ],
            # sentence 2
            [
                [1,0,0,1,1],
                [1,1,0,0,0],
                [0,1,1,1,0]
            ]
        ]
    )

    m = Metrics()
    m.set_metrics(labels, preds)
    print(m)
    m.update_metrics(labels, preds)
    print(m)