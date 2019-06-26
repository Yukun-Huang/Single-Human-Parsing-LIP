#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import getpass
import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.lip import LIP
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

parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
parser.add_argument('--data-path', type=str, help='Path to dataset folder')
parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
parser.add_argument('--batch-size', type=int, default=1, help="Number of images sent to the network in one step.")
parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")

parser.add_argument('-v', '--visualize', action='store_true', help="Display output and ground truth.")
args = parser.parse_args()


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
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


def show_image(img, pred, gt):

    fig, axes = plt.subplots(1, 3)
    ax0, ax1, ax2 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    classes = np.array(('Background',  # always index 0
                        'Hat',          'Hair',      'Glove',     'Sunglasses',
                        'UpperClothes', 'Dress',     'Coat',      'Socks',
                        'Pants',        'Jumpsuits', 'Scarf',     'Skirt',
                        'Face',         'Left-arm',  'Right-arm', 'Left-leg',
                        'Right-leg',    'Left-shoe', 'Right-shoe', ))
    colormap = [(0,0,0),
                (1,0.25,0), (0,0.25,0),  (0.5,0,0.25),   (1,1,1),
                (1,0.75,0), (0,0,0.5),   (0.5,0.25,0),   (0.75,0,0.25),
                (1,0,0.25), (0,0.5,0),   (0.5,0.5,0),    (0.25,0,0.5),
                (1,0,0.75), (0,0.5,0.5), (0.25,0.5,0.5), (1,0,0),
                (1,0.25,0), (0,0.75,0),  (0.5,0.75,0),                 ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img
    img = denormalize(img[0].numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.transpose(1,2,0).reshape((h,w,3))
    pred = pred.reshape((h,w))
    gt = gt.reshape((h, w))

    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)
    ax2.set_title('gt')
    mappable = ax2.imshow(gt, cmap=cmap, norm=norm)
    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=0.7,)  # orientation='horizontal'
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(classes):
        cbar.ax.text(1.3, (j+0.45) / 20.0, lab, ha='left', va='center',)  # fontsize=7
    # cbar.ax.get_yaxis().labelpad = 5

    plt.savefig(fname="result.jpg")
    plt.show()


def get_pixel_acc(pred, gt):
    valid = (gt >= 0)
    acc_sum = (valid * (pred == gt)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc


def get_mean_acc(pred, gt, numClass):
    imPred = pred.copy()
    imLabel = gt.copy()

    imPred += 1
    imLabel += 1
    imPred = imPred * (imLabel > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLabel)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area label:
    (area_label, _) = np.histogram(imLabel, bins=numClass, range=(1, numClass))

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    valid = area_label > 0

    # Compute intersection over union:
    classes_acc = area_intersection / (area_label + 1e-10)
    mean_acc = np.average(classes_acc, weights=valid)
    return mean_acc


def get_mean_IoU(pred, gt, numClass):
    imPred = pred.copy()
    imLabel = gt.copy()

    imPred += 1
    imLabel += 1
    imPred = imPred * (imLabel > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLabel)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_label, _) = np.histogram(imLabel, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_label - area_intersection

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    valid = area_label > 0

    # Compute intersection over union:
    IoU = area_intersection / (area_union + 1e-10)
    mean_IoU = np.average(IoU, weights=valid)
    return mean_IoU


def get_mean_acc_and_IoU(pred, gt, numClass):
    imPred = pred.copy()
    imLabel = gt.copy()

    imPred += 1
    imLabel += 1
    imPred = imPred * (imLabel > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLabel)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_label, _) = np.histogram(imLabel, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_label - area_intersection

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    valid = area_label > 0

    # Compute mean acc.
    classes_acc = area_intersection / (area_label + 1e-10)
    mean_acc = np.average(classes_acc, weights=valid)

    # Compute intersection over union:
    IoU = area_intersection / (area_union + 1e-10)
    mean_IoU = np.average(IoU, weights=valid)
    return mean_acc, mean_IoU


def main():
    # --------------- model --------------- #
    snapshot = os.path.join(args.models_path, args.backend, 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args.backend)
    net.eval()

    # ------------ data loader ------------ #
    data_transform = get_transform()
    val_loader = DataLoader(LIP(args.data_path, train=False, transform=data_transform['img'],
                                gt_transform=data_transform['gt']),
                            batch_size=args.batch_size,
                            shuffle=False,
                            )

    # --------------- eval --------------- #
    overall_acc_list = []
    mean_acc_list = []
    mean_IoU_list = []

    with torch.no_grad():
        for index, (img, gt) in enumerate(val_loader):
            pred_seg, pred_cls = net(img.cuda())
            pred_seg = pred_seg[0]
            pred = pred_seg.cpu().numpy().transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
            gt = np.asarray(gt.numpy(), dtype=np.uint8).transpose(1, 2, 0)

            if args.visualize:
                show_image(img, pred, gt)

            overall_acc_list.append(get_pixel_acc(pred, gt))
            _mean_acc, _mean_IoU = get_mean_acc_and_IoU(pred, gt, args.num_classes)
            mean_acc_list.append(_mean_acc)
            mean_IoU_list.append(_mean_IoU)
            
            print(' %d / %d ' % (index, len(val_loader)))

    print(' overall acc. : %f ' % (np.mean(overall_acc_list)))
    print(' mean acc.    : %f ' % (np.mean(mean_acc_list)))
    print(' mean IoU     : %f ' % (np.mean(mean_IoU_list)))


if __name__ == '__main__':
    main()
