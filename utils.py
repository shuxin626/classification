import os
import torch
import matplotlib.pyplot as plt
import glob


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class CkptController(object):
    def __init__(self, train_param, clean_prev_ckpt_flag=True, ckpt_dir=None, dir_name_suffix='') -> None:
        self.dir_name_suffix = dir_name_suffix
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
        else:
            self.ckpt_dir = self.create_ckpt_dir_handle(train_param)
        if clean_prev_ckpt_flag:
            clean_pt_files_in_dir(self.ckpt_dir)
        print('ckpt dir is {}'.format(self.ckpt_dir))

    def create_ckpt_dir_handle(self, train_param):
        ckpt_dir = 'checkpoint/{}-{}-{}'.format(train_param['net'],
                                                train_param['batch_size'], 
                                                train_param['lr'])
        ckpt_dir = ckpt_dir + self.dir_name_suffix
        cond_mkdir(ckpt_dir)
        return ckpt_dir

    def save_ckpt(self, model, train_acc, val_acc, epoch):
        state = {
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'epoch': epoch,
            }
        torch.save(state, os.path.join(self.ckpt_dir, '{}.pth'.format(epoch)))

    def load_ckpt(self, ckpt_num=None):
        if ckpt_num is None:
            filelist = glob.glob(os.path.join(self.ckpt_dir, "*.pth"))
            assert filelist != [], 'dir is empty and we need to create one'
            ckpt_dict = (torch.load(sort_file_by_digit_in_name(filelist)[-1], map_location=device))
        else:
            ckpt_dict = (torch.load(os.path.join(self.ckpt_dir, '{}.pth'.format(ckpt_num)), map_location=device))
        return ckpt_dict

def sort_file_by_digit_in_name(filelist, suffix='.pth'):
    end_index = len(suffix)
    file_name_list_int = [int(os.path.basename(file)[:-end_index]) for file in filelist]
    folder_name = os.path.dirname(filelist[0])
    sortted_file_name_list_int = sorted(file_name_list_int)
    sortted_file_name_list_str = [os.path.join(folder_name, '{}{}'.format(num ,suffix)) for num in sortted_file_name_list_int]
    return sortted_file_name_list_str

    
def plot_loss(iter, loss, filename="loss", label=None, newfig=False, color="b"):
    plt.figure(1)
    plt.clf()
    plt.title(filename)
    plt.xlabel("epoch")
    # plt.ylabel("loss")
    _ = plt.plot(iter, loss)
    if newfig:
        if filename is not None:
            plt.savefig('imgs/'+filename + ".png",
                        dpi=200, bbox_inches="tight")
    plt.draw()
    plt.show()
    
def clean_pt_files_in_dir(path_to_dir):
    # https://www.techiedelight.com/delete-all-files-directory-python/
    filelist = glob.glob(os.path.join(path_to_dir, "*.pth"))
    for f in filelist:
        os.remove(f)