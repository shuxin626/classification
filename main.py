from ideal_classfication_trainer import IdealClassificationTrainer
from wbcdataset import dataio
from config import *
from resnet_torch import *

train_param = {
    'optimizer': 'sgd', # ['sgd', 'adm']
    'lr': 0.001,
    'batch_size': 32,
    'val_ratio': 0.2, 
    'net': 'unet10',
    'training_epochs': 1000,
    'early_stop_epochs': 100,
    'early_stop_metrics': 'val_loss',
    'checkpoint': {
        'save_checkpoint': True,
        'clean_prev_ckpt_flag': True,
        'dir_name_suffix': '',
        'metrics': 'val_loss',
    },
    'data_folder': 'D:/OneDrive_1_10-12-2022/',
}

train_loader, val_loader, test_loader = dataio(train_param['data_folder'], train_param['batch_size'], shuffle_data=True)
if train_param['net'] == 'unet10':
    model = ResNet10(1, 4).to(device)
trainer = IdealClassificationTrainer(model, train_param)

trainer.fit(train_loader, val_loader)