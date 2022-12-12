#%%
from ideal_classfication_trainer import IdealClassificationTrainer
from tester import ClassificationTester
from wbcdataset import dataio
from config import *
from resnet_torch import *

mode = 'train' # from ['train', 'test', 'grad_cam']

train_param = {
    'optimizer': 'sgd', # ['sgd', 'adam']
    'lr': 0.0001,
    'batch_size': 32,
    'val_num_per_type': 0.2, 
    'shuffle_data': True,
    'net': 'unet10',
    'training_epochs': 1000,
    'early_stop_epochs': 30,
    'early_stop_metrics': 'val_loss',
    'checkpoint': {
        'save_checkpoint': True,
        'clean_prev_ckpt_flag': True,
        'dir_name_suffix': '-bg-0',
        'metrics': 'val_loss',
    },
    'data_folder': 'D:/OneDrive_1_10-12-2022/',
}

eval_param = {
    'dataset_for_test': ['val'],
    'ckpt_dir': 'checkpoint/unet10-32-0.0001',
    'ckpt_num': None,
}

train_loader, val_loader, test_loader, type_count = dataio(train_param['data_folder'], train_param['batch_size'], shuffle_data=train_param['shuffle_data'], val_num_per_type=train_param['val_num_per_type'])
if train_param['net'] == 'unet10':
    model = ResNet10(1, type_count).to(device)


if mode == 'train':
    trainer = IdealClassificationTrainer(model, train_param)
    trainer.fit(train_loader, val_loader)
elif mode == 'test':
    tester = ClassificationTester(model, eval_param['ckpt_dir'], eval_param['ckpt_num'], eval_param['dataset_for_test'])
    tester.fit(train_loader, val_loader, test_loader)
elif mode == 'grad_cam':
    tester = ClassificationTester(model, eval_param['ckpt_dir'], eval_param['ckpt_num'], eval_param['dataset_for_test'])
    tester.grad_cam(val_loader)