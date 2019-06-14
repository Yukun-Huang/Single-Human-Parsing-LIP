#!/usr/local/bin/python3
import os
import argparse
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.lip import LIPWithClass
from net.pspnet import PSPNet

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

parser = argparse.ArgumentParser(description="Human Parsing")
parser.add_argument('--data-path', type=str, help='Path to dataset folder')
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
parser.add_argument('--snapshot', type=str, default=None, help='Path to pre-trained weights')
parser.add_argument('--batch-size', type=int, default=16, help="Number of images sent to the network in one step.")
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs to run')
parser.add_argument('--crop_x', type=int, default=256, help='Horizontal random crop size')
parser.add_argument('--crop_y', type=int, default=256, help='Vertical random crop size')
parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
parser.add_argument('--start-lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
args = parser.parse_args()


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        print("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_gt_list = [
        transforms.Resize((256, 256), 0),
        transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
    ]

    data_transforms = {
        'img': transforms.Compose(transform_image_list),
        'gt': transforms.Compose(transform_gt_list),
    }
    return data_transforms


def get_dataloader():
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes,
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    data_transform = get_transform()
    train_loader = DataLoader(LIPWithClass(root=args.data_path, transform=data_transform['img'],
                                           gt_transform=data_transform['gt']),
                              batch_size=args.batch_size,
                              shuffle=True,
                              )
    return train_loader


if __name__ == '__main__':

    models_path = os.path.join('./checkpoints', args.backend)
    os.makedirs(models_path, exist_ok=True)

    train_loader = get_dataloader()

    net, starting_epoch = build_network(args.snapshot, args.backend)
    optimizer = optim.Adam(net.parameters(), lr=args.start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in args.milestones.split(',')])

    for epoch in range(1+starting_epoch, 1+starting_epoch+args.epochs):
        seg_criterion = nn.NLLLoss(weight=None)
        cls_criterion = nn.BCEWithLogitsLoss(weight=None)
        epoch_losses = []
        net.train()

        for count, (x, y, y_cls) in enumerate(train_loader):
            # input data
            x, y, y_cls = x.cuda(), y.cuda().long(), y_cls.cuda().float()
            # forward
            out, out_cls = net(x)
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + args.alpha * cls_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print
            epoch_losses.append(loss.item())
            status = '[{0}] step = {1}/{2}, loss = {3:0.4f} avg = {4:0.4f}, LR = {5:0.7f}'.format(
                epoch, count, len(train_loader),
                loss.item(), np.mean(epoch_losses), scheduler.get_lr()[0])
            print(status)

        scheduler.step()
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch)])))

    torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", 'last'])))
