import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import numpy as np


def dataio(folder_name, batch_size, shuffle_data, val_ratio=0.2):
    dataset_train = WBCDataSet(train=True, folder_name=folder_name)
    dataset_test = WBCDataSet(train=False, folder_name=folder_name)
   
    dataset_train_indices = list(range(len(dataset_train)))
    if shuffle_data:
        np.random.shuffle(dataset_train_indices)
    val_split_index = int(np.floor(val_ratio * len(dataset_train)))
    train_indices, val_indices = dataset_train_indices[val_split_index:], \
                                 dataset_train_indices[:val_split_index]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
    
    
    
    
class WBCDataSet(Dataset):
    def __init__(self, train, folder_name):
        self.dataset, self.targetset = self.load_data_from_pickle(train, folder_name)
        self.type_count = np.max(self.targetset) + 1
        self.img_width = self.dataset.shape[1]
        self.img_height=self.dataset.shape[2]
       
    def load_data_from_pickle(self, train, file):

        if train: 
            with open(file + 'train_data_set.pickle', 'rb') as input1:
                dataset = pickle.load(input1)
            with open(file + 'train_target_index.pickle', 'rb') as input2:
                targetset = pickle.load(input2)

        else:
            with open(file + 'test_data_set.pickle', 'rb') as input3:
                dataset = pickle.load(input3)
            with open(file + 'test_target_set.pickle', 'rb') as input4:
                targetset = pickle.load(input4)

        targetset = np.array(targetset)
        targetset = targetset - 1

        return dataset, targetset

    def __len__(self):
        return len(self.targetset)

    def __getitem__(self, index):
        image = self.dataset[index, ...]
        image = torch.tensor(image)
        image = torch.permute(image, [2, 0, 1])
        target = self.targetset[index]
        return image, target
        
        
if __name__ == '__main__':
    dataset = WBCDataSet(True, 'D:/OneDrive_1_10-12-2022/')
    img, target = dataset[2]
    print('x')
