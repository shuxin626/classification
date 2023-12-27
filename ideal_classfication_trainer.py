
from utils import CkptController, plot_loss
from config import *
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau



class IdealClassificationTrainer():
    def __init__(self, model, train_param, data_param):
        super(IdealClassificationTrainer, self).__init__()

        self.model = model
        if train_param['pretrained']['load_pretrained']:
            ckpt_controller = CkptController(None, data_param, clean_prev_ckpt_flag=False, ckpt_dir=train_param['pretrained']['ckpt_dir'])
            self.ckpt_state = ckpt_controller.load_ckpt(train_param['pretrained']['ckpt_num'])
            ckpt_state_dict = {name: weights for name, weights in self.ckpt_state['state_dict'].items() if 'linear' not in name}
            self.model.load_state_dict(ckpt_state_dict, strict=False)

           
        if train_param['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=train_param['lr'], momentum=0.5)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10, min_lr=train_param['lr']*0.01)
        elif train_param['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

        else:
            raise Exception('Optimizer not found')

        self.criterion = nn.CrossEntropyLoss()
        self.train_param = train_param

        if self.train_param['checkpoint']['save_checkpoint']:
            self.ckpt_controller = CkptController(
                train_param, data_param, self.train_param['checkpoint']['clean_prev_ckpt_flag'], dir_name_suffix=self.train_param['checkpoint']['dir_name_suffix'])
   
    
    def summarize_result(self, epoch, result_epoch, result_lst, best_result, early_stop_counter, save_ckpt,
                         ckpt_metrics, early_stop_metrics, key_list=['train_loss', 'train_acc', 'val_loss', 'val_acc']):

        for key in key_list:
            result_lst[key].append(result_epoch[key])

            if 'loss' in key and result_epoch[key] < best_result[key]:
                best_flag = True
            elif 'loss' not in key and result_epoch[key] > best_result[key]:
                best_flag = True
            else:
                best_flag = False

            if best_flag:
                best_result[key] = result_epoch[key]
                
                if save_ckpt and key == ckpt_metrics:
                    self.ckpt_controller.save_ckpt(
                        self.model, result_epoch['train_acc'], result_epoch['val_acc'], epoch)
                    print('saving ckpt at epoch {} according to the {}'.format(epoch, self.train_param['checkpoint']['metrics']))
                    
                if key == early_stop_metrics:
                    early_stop_counter = 0
            elif key == early_stop_metrics and not best_flag:
                early_stop_counter += 1
            

        print('{} is {}, early_stop_counter {}'.format(early_stop_metrics, result_epoch[early_stop_metrics], early_stop_counter))        

        return result_lst, best_result, early_stop_counter


    def train(self, epoch, train_loader):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if len(inputs) < self.train_param['batch_size']:
                break
            inputs = inputs.float().to(device)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device)
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0 or batch_idx == (len(train_loader) - 1):
                print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    (batch_idx + 1)* len(inputs),
                    len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader),
                    loss.item())
                )
        
        acc = 100.*correct/total
        print("avg train acc of epoch : %3d is : %3.4f" % (epoch, acc))
        return acc

    def test(self, epoch, val_loader, is_train_set=False):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.float().to(device)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        
        if is_train_set:
            print("train set acc of epoch : %3d is : %3.4f" % (epoch, acc))
        else:
            print("val acc of epoch : %3d is : %3.4f" % (epoch, acc))

        return test_loss, acc

    def fit(self, train_loader, val_loader):
        result_epoch = {}
        result_lst = {'train_loss': [], 'train_acc': [],
                      'val_loss': [], 'val_acc': []}
        best_result = {'train_loss': 1000, 'train_acc': 0,
                       'val_loss': 1000, 'val_acc': 0}
        epoch_lst = []
        early_stop_counter = 0

        for epoch in range(self.train_param['training_epochs']):
            epoch_lst.append(epoch)

            train_acc_avg = self.train(
                epoch, train_loader)

            result_epoch['val_loss'], result_epoch['val_acc'] = self.test(
                epoch, val_loader)
            result_epoch['train_loss'], result_epoch['train_acc'] = self.test(epoch, train_loader, True)
            
            if self.train_param['optimizer'] == 'sgd':
                self.scheduler.step(result_epoch['train_loss'])
            
            result_lst, best_result, early_stop_counter = self.summarize_result(epoch, result_epoch, result_lst, best_result, early_stop_counter,self.train_param['checkpoint']['save_checkpoint'], self.train_param['checkpoint']['metrics'], self.train_param['early_stop_metrics'])

            if early_stop_counter == self.train_param['early_stop_epochs']:
                break

            if (epoch % 15 == 0) and epoch > 1 or (epoch == self.train_param['training_epochs']-1):
                plot_loss(epoch_lst, result_lst['val_acc'], 'val_acc')
                plot_loss(epoch_lst, result_lst['train_acc'], 'train_acc')
                plot_loss(epoch_lst, result_lst['train_loss'], 'train_loss')
                plot_loss(epoch_lst, result_lst['val_loss'], 'val_loss')

            print("best train_acc" + " is : %3.5f" %
                  (best_result['train_acc']))
            print("best val_acc" + " is : %3.5f" % (best_result['val_acc']))
            result_epoch = {}
