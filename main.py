import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import time

from data import VOCDetection
from model import Yolo
from model import Loss
from utils import NMS
from utils import inference
from utils import decoder

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

root = '/home/woojin/yolo'
ckpt_root = 'checkpoints'   # from/to which directory to load/save checkpoints.
data_root = 'dataset'       # where the data exists.
pretrained_backbone_path = 'weights/vgg_features.pth'

DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'

ckpt_dir = os.path.join(root, ckpt_root)
makedirs(ckpt_dir)

def main():
    # Configurations
    lr = 0.00000001          # learning rate
    batch_size = 64     # batch_size
    last_epoch = 1      # the last training epoch. (defulat: 1)
    max_epoch = 553   # maximum epoch for the training.

    num_boxes = 2       # the number of boxes for each grid in Yolo v1.
    num_classes = 20    # the number of classes in Pascal VOC Detection.
    grid_size = 7       # 3x224x224 image is reduced to (5*num_boxes+num_classes)x7x7.
    lambda_coord = 7    # weight for coordinate regression loss.
    lambda_noobj = 0.5  # weight for no-objectness confidence loss.

    """ dataset load """
    train_dset = VOCDetection(root=data_root, split='train')
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    #drop_last 마지막 애매하게 남는 데이터들은 버림
    test_dset = VOCDetection(root=data_root, split='test')
    test_dloader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)

    """ model load """
    model = Yolo(grid_size, num_boxes, num_classes)
    #model = nn.DataParallel(model, device_ids = [5,6,7])
    model = model.to(DEVICE)
    
    #pretrained_weights = torch.load(pretrained_backbone_path)
    #model.load_state_dict(pretrained_weights)
    
    """ optimizer / loss """
    model.features.requires_grad_(False)
    model_params = [v for v in model.parameters() if v.requires_grad is True]
    optimizer = optim.Adam(model_params, lr=lr, betas=[0.9,0.999])
    # Load the last checkpoint if exits.
    ckpt_path = os.path.join(ckpt_dir, 'last_best.pth') 
    if os.path.exists(ckpt_path): 
        ckpt = torch.load(ckpt_path, map_location='cuda:3')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        last_epoch = ckpt['epoch'] + 1
        print('Last checkpoint is loaded. start_epoch:', last_epoch)
    else:
        print('No checkpoint is found.')


    Yolov1Loss = Loss(7,2,20)
    #ckpt_path = os.path.join(ckpt_dir, 'last_best.pth')
    """ training """
    # Training & Testing.
    model = model.to(DEVICE)
    best_loss = 1
    for epoch in range(1, max_epoch):
        step = 0
        # Learning rate scheduling
        if epoch in [50,150,550,600]:
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch < last_epoch:
            continue
        
        model.train()
        for x, y in train_dloader:
            step += 1
            imgs = Variable(x)
            gt_outs = Variable(y)
            imgs, gt_outs = imgs.to(DEVICE), gt_outs.to(DEVICE)
            model_outs = model(imgs)
            loss = Yolov1Loss(model_outs, gt_outs)
            
            if loss < best_loss:
                best_loss = loss
                ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch}
                torch.save(ckpt, ckpt_path)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('step:{}/{} | loss:{:.8f}'.format(step,len(train_dloader), loss.item()))
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_dloader:
                imgs = Variable(x)
                gt_outs = Variable(y)
                imgs, gt_outs = imgs.to(DEVICE), gt_outs.to(DEVICE)

                model_outs = model(imgs)
                loss = Yolov1Loss(model_outs, gt_outs)
                loss_iter = loss.item()
            print('Epoch [%d/%d], Val Loss: %.4f' % (epoch, max_epoch, loss_iter))
        
        ckpt = {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch}
        torch.save(ckpt, ckpt_path)
        

    ''' test '''
    
    test_image_dir = os.path.join(root,'test_images')
    image_path_list = [os.path.join(test_image_dir, path) for path in os.listdir(test_image_dir)]

    for image_path in image_path_list:
        inference(model, image_path)
    

if __name__ == '__main__':
    main()