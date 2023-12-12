from math import ceil
from config import *
from utils import CkptController
import numpy as np
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pickle
import pandas as pd
from drawing import scatter3d_draw, color_dict

class ClassificationTester(object):

    def __init__(self, model, ckpt_dir, ckpt_num, dataset_for_test, tsne_param):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        ckpt_controller = CkptController(None, clean_prev_ckpt_flag=False, ckpt_dir=ckpt_dir)
        self.ckpt_state = ckpt_controller.load_ckpt(ckpt_num)
        self.model.load_state_dict(self.ckpt_state['state_dict'])
        print('loaded model has train accuracy {}'.format(self.ckpt_state['train_acc']))
        print('loaded model has val accuracy {}'.format(self.ckpt_state['val_acc']))
        self.dataset_for_test = dataset_for_test
        self.tsne_param = tsne_param

    def test(self, dataloader):
        # topest_mask_ind is the ind of the best mask in the batch of maskquery
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        # for subbatch mode
        outputs_accum = torch.tensor([]).to(device)
        feature_all = torch.tensor([])
        targets_all = torch.tensor([])
        
        with torch.no_grad():
            for iter_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.float().to(device)
                targets_all = torch.cat((targets_all, targets))
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device)
                
                outputs, feature_batch = self.model(inputs)
              
                loss = self.criterion(outputs, targets)
                # test_loss = test_loss + loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                feature_all = torch.cat((feature_all, feature_batch.cpu()), dim=0)

                
                if (iter_idx + 1) % 5 == 0:
                    print('[{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                        (iter_idx + 1)* len(inputs),
                        len(dataloader.dataset),
                        100. * iter_idx / len(dataloader),
                        loss.item())
                    )

        acc = 100.*correct/total
        return acc, [feature_all, targets_all]
    
    def grad_cam(self, dataloader):
        self.model.eval()        
        inputs_batch, targets_batch = next(iter(dataloader))
        for ith_in_batch in range(inputs_batch.size(dim=0)):
            inputs = inputs_batch[ith_in_batch, ...][None, ...]
            targets = targets_batch[ith_in_batch]
            
            inputs = inputs.float().to(device)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device)
            
            outputs, _ = self.model(inputs, cal_grad_cam=True)
            _, predicted = outputs.max(1)
            if predicted == targets: 
                print('predict correct')
            else:
                print('predict wrong')
                
            loss = self.criterion(outputs, predicted)
            loss.backward()
            gradients = self.model.get_activations_gradient()
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            activations = self.model.get_activations(inputs).detach()
            for i in range(pooled_gradients.size(dim=0)):
                activations[:, i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = F.interpolate(heatmap[None, None, ...], size=(300, 300))
            # heatmap_max = heatmap.max(axis=0)[0]
            # heatmap /= heatmap_max
            plt.imshow(heatmap[0, 0].detach().cpu())
            plt.colorbar()
            plt.show()
            plt.imshow(inputs[0, 0, ...].detach().cpu())
            plt.colorbar()
            plt.show()
        
                

    def fit(self, train_loader, val_loader, test_loader):

        if 'train' in self.dataset_for_test: 
            print('test train dataset')
            train_acc, feature_and_target_all = self.test(train_loader)
            print("exp train acc is : %3.4f" % (train_acc))    
        if 'val' in self.dataset_for_test: 
            print('test val dataset')
            val_acc, feature_and_target_all = self.test(val_loader)
            print("exp val acc is : %3.4f" % (val_acc))
        if 'test' in self.dataset_for_test:
            print('test test dataset')
            test_acc, feature_and_target_all = self.test(test_loader)
            print("exp test acc is : %3.4f" % (test_acc))
            
        if self.tsne_param['cal_tsne'] == True:
            feature_all = feature_and_target_all[0].numpy()
            targets_all = feature_and_target_all[1].numpy()
            tsne = TSNE(n_components=3, random_state=42)
            tsne_results = tsne.fit_transform(feature_all)
            df_tsne = pd.DataFrame(tsne_results, columns=['x', 'y', 'z'])
            df_tsne['type'] = targets_all.astype(np.int8)
            if self.tsne_param['path_to_save_data'] is not None:
                df_tsne.to_csv(self.tsne_param['path_to_save_data'], index=False)
            if self.tsne_param['draw_figure']:
                scatter3d_draw(df_tsne, ['Monocyte', 'Granulocyte', 'Lymphocyte'])

                
                
            
            
