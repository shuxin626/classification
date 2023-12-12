#%%
from ideal_classfication_trainer import IdealClassificationTrainer
from tester import ClassificationTester
from wbcdataset import dataio
from config import *
from resnet_torch import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

mode = 'test' # from ['train', 'test', 'grad_cam']

train_param = {
    'optimizer':'sgd', # ['sgd', 'adam']
    'lr': 0.0005,
    'batch_size': 64,
    'type-str': 'b-t-(m-g)',
    'val_num_per_type': 100, 
    'shuffle_data': True,
    'net': 'unet10',
    'training_epochs': 1000,
    'early_stop_epochs': 30,
    'early_stop_metrics': 'val_loss',
    'checkpoint': {
        'save_checkpoint': True,
        'clean_prev_ckpt_flag': True,
        'dir_name_suffix': '',
        'metrics': 'val_loss',
    },
    'data_folder': '../clean_dataset/',
    'pretrained': {
        'load_pretrained':  False,  # IMPORTANT: whether startover or from prev one
        # from ckpt_dir to load checkpoint
        'ckpt_dir': '',
        'ckpt_num': None,  # int
    }
    }

eval_param = {
    'dataset_for_test': ['test'],
    'ckpt_dir': 'checkpoint/unet10-64-0.0005',
    'ckpt_num': None,
    'tsne_param': {
        'cal_tsne': True,
        'path_to_save_data': 'tsne_data/test_tsne_and_targets.csv',
        'draw_figure': True,
    }
}

train_loader, val_loader, test_loader, type_count = dataio(train_param['data_folder'], train_param['batch_size'], 
                                                           shuffle_data=train_param['shuffle_data'], 
                                                           val_num_per_type=train_param['val_num_per_type'],
                                                           type_str=train_param['type-str'])
if train_param['net'] == 'unet10':
    model = ResNet10(1, type_count).to(device)


if mode == 'train':
    trainer = IdealClassificationTrainer(model, train_param)
    trainer.fit(train_loader, val_loader)
elif mode == 'test':
    tester = ClassificationTester(model, eval_param['ckpt_dir'], eval_param['ckpt_num'], eval_param['dataset_for_test'], eval_param['tsne_param'])
    tester.fit(train_loader, val_loader, test_loader)
elif mode == 'grad_cam':
    tester = ClassificationTester(model, eval_param['ckpt_dir'], eval_param['ckpt_num'], eval_param['dataset_for_test'], eval_param['tsne_param'])
    tester.grad_cam(val_loader)