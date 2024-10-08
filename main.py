#%%
from ideal_classfication_trainer import IdealClassificationTrainer
from tester import ClassificationTester
from cascaded_tester import CascadedClassificationTester
from wbcdataset import dataio
from config import *
from resnet_torch import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_ckpt_for_eval

mode = 'test' # from ['train', 'test', 'grad_cam']

data_param = {
    'type-str': 'm-g-b-t', # ['b-t', 'm-g-(b-t)', 'm-g-b-t']
    'val_num_per_type': 100, 
    'shuffle_data': False,
    'data_folder': '../direct_cut/',
}

train_param = {
    'optimizer':'adam', # ['sgd', 'adam']
    'lr': 0.00008, # adam 0.00008 for three types aug
    'batch_size': 16,
    'net': 'unet10',
    'training_epochs': 1000,
    'early_stop_epochs': 15,
    'early_stop_metrics': 'val_loss',
    'checkpoint': {
        'save_checkpoint': True,
        'clean_prev_ckpt_flag': False,
        'dir_name_suffix': 'aug',
        'metrics': 'val_loss',
    },
    'pretrained': {
        'load_pretrained': False, # for fine-tune b-t classifier
        'ckpt_dir': 'checkpoint/unet10-16-8e-05-m-g-(b-t)aug',
        'ckpt_num': None,  # int
    }
    }

eval_param = {
    'dataset_for_test': ['test'],
    'ckpt_dir': 'checkpoint/unet10-16-8e-05-m-g-b-taug',
    'ckpt_num': None,
    'cascade_param':
        {
            'if_cascade': True,
            'ckpt_dir': 'checkpoint/unet10-16-0.0001-b-t',
            'further_classify_which_type_in_first_model': 2, # int
        },
    'tsne_param': { # only take effect when 'if_cascade' = False
        'cal_tsne': True,
        'path_to_save_data': 'tsne_data/test_tsne_and_targets.csv',
        'draw_figure': True,
    },
    'draw_prc': True,
}

train_loader, val_loader, test_loader, type_count = dataio(data_param['data_folder'], train_param['batch_size'], 
                                                           shuffle_data=data_param['shuffle_data'], 
                                                           val_num_per_type=data_param['val_num_per_type'],
                                                           type_str=data_param['type-str'])
if train_param['net'] == 'unet10': 
    if mode=='train':
        model = ResNet10(1, type_count).to(device)
    elif mode == 'test' and eval_param['cascade_param']['if_cascade'] == False:
        model = ResNet10(1, type_count).to(device)
        model = load_ckpt_for_eval(eval_param['ckpt_dir'], eval_param['ckpt_num'], model)
    else:
        # only cover firstly classify m-g-(b-t) and then classify b-t
        model1 = ResNet10(1, 3).to(device)
        model1 = load_ckpt_for_eval(eval_param['ckpt_dir'], eval_param['ckpt_num'], model1)
        model2 = ResNet10(1, 2).to(device)
        model2 = load_ckpt_for_eval(eval_param['cascade_param']['ckpt_dir'], None, model2)

if mode == 'train':
    trainer = IdealClassificationTrainer(model, train_param, data_param)
    trainer.fit(train_loader, val_loader)
elif mode == 'test' and eval_param['cascade_param']['if_cascade'] == False:
    tester = ClassificationTester(model, eval_param['dataset_for_test'], eval_param, data_param['type-str'])
    tester.fit(train_loader, val_loader, test_loader)
elif mode == 'test' and eval_param['cascade_param']['if_cascade'] == True:
    tester = CascadedClassificationTester(model1, model2, eval_param['dataset_for_test'], eval_param['cascade_param']['further_classify_which_type_in_first_model'])
    tester.fit(train_loader, val_loader, test_loader)
elif mode == 'grad_cam':
    tester = ClassificationTester(model, eval_param['dataset_for_test'], eval_param, data_param['type-str'])
    tester.grad_cam(val_loader)