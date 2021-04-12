import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        '''
        Assumes directory structure
        ../LR/LRBI/<testset>/x<scale>/name_LRBI_x<scale>.png
        ../HR/<testset>/x<scale>/name_HR_x<scale>.png
        '''
        apath = args.testpath + '/' + args.testset + '/x' + str(args.scale[0])
        hrpath = args.testpath + '/../../HR/' + args.testset + '/x' + str(args.scale[0])

        self.filelist = []
        self.hr_filelist = []
        self.imnamelist = []
        if not train:
            for f in os.listdir(apath):
                try:
                    # Read LR image
                    filename = os.path.join(apath, f)
                    misc.imread(filename)
                    self.filelist.append(filename)
                    self.imnamelist.append(f)

                    # Read HR image
                    prefix, _, postfix = f.split('_')
                    hr_f = prefix + '_HR_' + postfix
                    hr_filename = os.path.join(hrpath, hr_f)
                    misc.imread(hr_filename)
                    self.hr_filelist.append(hr_filename) 
                except:
                    pass

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]
        lr_tensor = common.np2Tensor([lr], self.args.rgb_range)[0]

        hr_filename = os.path.split(self.hr_filelist[idx])[-1]
        hr_filename, _ = os.path.splitext(hr_filename)
        hr = misc.imread(self.hr_filelist[idx])
        hr = common.set_channel([hr], self.args.n_colors)[0]
        hr_tensor = common.np2Tensor([hr], self.args.rgb_range)[0]

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

