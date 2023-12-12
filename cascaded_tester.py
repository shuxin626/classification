from math import ceil
from config import *
from utils import CkptController
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pandas as pd

class CascadedClassificationTester(object):

    def __init__(self, model1, model2, dataset_for_test, further_classify_which_type_in_first_model):
        self.model1 = model1
        self.model2 = model2
        self.criterion = nn.CrossEntropyLoss()
        self.dataset_for_test = dataset_for_test
        self.further_classify_which_type_in_first_model = further_classify_which_type_in_first_model

    def test(self, dataloader):
        self.model1.eval()
        self.model2.eval()
        correct = 0
        total = 0
        # for subbatch mode
        outputs_accum = torch.tensor([]).to(device)
        
        with torch.no_grad():
            for iter_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.float().to(device)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device)
                
                outputs, _ = self.model1(inputs)
                _, predicted_labels1 = outputs.max(1)

                # Now for inputs where model1 predicts the xx class, use model2
                mask = (predicted_labels1 == self.further_classify_which_type_in_first_model)                 
                if mask.any():
                    outputs2, _ = self.model2(inputs[mask])
                    _, predicted_labels2 = torch.max(outputs2, 1)
                    predicted_labels1[mask] = predicted_labels2 + self.further_classify_which_type_in_first_model  # Offset
                
                total += targets.size(0)
                correct += predicted_labels1.eq(targets).sum().item()

                
                if (iter_idx + 1) % 5 == 0:
                    print('[{:6}/{:6} ({:3.0f}%)]'.format(
                        (iter_idx + 1)* len(inputs),
                        len(dataloader.dataset),
                        100. * iter_idx / len(dataloader))
                    )

        acc = 100.*correct/total
        return acc
    

    def fit(self, train_loader, val_loader, test_loader):

        if 'train' in self.dataset_for_test: 
            print('test train dataset')
            train_acc = self.test(train_loader)
            print("exp train acc is : %3.4f" % (train_acc))    
        if 'val' in self.dataset_for_test: 
            print('test val dataset')
            val_acc = self.test(val_loader)
            print("exp val acc is : %3.4f" % (val_acc))
        if 'test' in self.dataset_for_test:
            print('test test dataset')
            test_acc = self.test(test_loader)
            print("exp test acc is : %3.4f" % (test_acc))
            
