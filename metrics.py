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
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.f1_score = []
        self.true_positives = []
        self.true_negatives = []
        self.false_positives = []
        self.false_negatives = []
        self.num_examples = []

        num_categories = labels.shape[-1]

        for i in range(num_categories):
            tp, tn, fp, fn = self.get_positives_and_negatives(labels[:,:,i], predictions[:,:,i])

            self.true_positives += [tp]
            self.true_negatives += [tn]
            self.false_positives += [fp]
            self.false_negatives += [fn]
            self.num_examples += [tp+tn+fp+fn]

            precision = self.get_precision(tp, fp)
            recall = self.get_recall(tp,fn)
            acc = self.get_accuracy(tp, tn, fp, fn)
            f1_score = self.get_f1_score(precision, recall)
            self.precision += [precision]
            self.recall += [recall]
            self.accuracy += [acc]
            self.f1_score += [f1_score] 

    class InconsistentClassesError(Exception):
        def __init__(self, old_num_classes, new_num_classes) -> None:
            msg = f"Inconsistent number of classes: {old_num_classes}, {new_num_classes}"
            super().__init__(msg)

    
    def update_metrics(self, labels, predictions):
        '''  
        @params:
        - labels: (d,n,num_categories) dimensional vector of 1s and 0s
        - predictions: (d,n,num_categories) dimensional vector of 1s and 0s
        '''
        labels = labels.contiguous()
        predictions = predictions.contiguous()

        try:
            try:
                if len(self.accuracy)!=labels.shape[-1]:
                    raise Metrics.InconsistentClassesError(len(self.accuracy), labels.shape[-1])
            except AttributeError:
                pass

            num_categories = labels.shape[-1]
            
            for i in range(num_categories):
                tp, tn, fp, fn = self.get_positives_and_negatives(labels[:,:,i], predictions[:,:,i])

                self.true_positives[i] += tp
                self.true_negatives[i] += tn
                self.false_positives[i] += fp
                self.false_negatives[i] += fn
                self.num_examples[i] = tp+tn+fp+fn

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
        except:
            self.set_metrics(labels, predictions)


    def get_positives_and_negatives(self, labels, predictions):
        '''  
        @params:
        - labels: (d,n,) dimensional vector of 1s and 0s
        - predictions: (d,n,) dimensional vector of 1s and 0s

        @function:
        - get true positives, true negatives, false positives, false negatives for ith class
        '''
        target = labels.view(-1)
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


    def get_accuracy(self, tp, tn, fp, fn):
        return round((tp+tn)/(tp+tn+fp+fn), self.n_decimals)


    def get_f1_score(self,precision, recall):
        if precision == 0 or recall == 0:
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
            ["num_examples"] + self.num_examples,
            ["acc"] + self.accuracy,
            ["precision"] + self.precision,
            ["recall"] + self.recall,
            ["f1_score"] + self.f1_score
        ]
        
        return PrettyPrint.get_tabular_formatted_string(
                    dataset=dataset, 
                    headers=headers,
                    include_serial_numbers=False,
                    table_header="Evaluation metrics",
                    partitions=[5,7]
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
    m.update_metrics(labels, preds)
    print(m)
    m.update_metrics(labels, preds)
    print(m)