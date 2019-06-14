from os import path as osp
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np


class LIP(data.Dataset):

    def __init__(self, root, train=True, transform=None, gt_transform=None ):
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.train = train  # trainval set or val set

        if self.train:
            self.train_image_path, self.train_gt_path = self.read_labeled_image_list(osp.join(root, 'train'))
        else:
            self.val_image_path, self.val_gt_path = self.read_labeled_image_list(osp.join(root, 'val'))
            # self.test_image_path = self.read_labeled_image_list(osp.join(root, 'test'))

    def __getitem__(self, index):
        if self.train:
            img, gt = self.get_a_sample(self.train_image_path, self.train_gt_path, index)
        else:
            img, gt = self.get_a_sample(self.val_image_path, self.val_gt_path, index)
        return img, gt

    def __len__(self):
        if self.train:
            return len(self.train_image_path)
        else:
            return len(self.val_image_path)

    def get_a_sample(self, image_path, gt_path, index):
        # get PIL Image
        img = Image.open(image_path[index])  # .resize((512,512),resample=Image.BICUBIC)
        if len(img.getbands()) != 3:
            img = img.convert('RGB')
        gt = Image.open(gt_path[index])  # .resize((30,30),resample=Image.NEAREST)
        if len(gt.getbands()) != 1:
            gt = gt.convert('L')

        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return img, gt

    def read_labeled_image_list(self, data_dir):
        # return img path list and groundtruth path list
        f = open(osp.join(data_dir, 'id.txt' ), 'r')
        image_path = []
        gt_path = []
        for line in f:
            image = line.strip("\n")
            if self.train:
                image_path.append(osp.join(data_dir, 'image', image + ".jpg"))
                gt_path.append(osp.join(data_dir, 'gt', image + ".png"))
            else:
                image_path.append(osp.join(data_dir, 'image', image + ".jpg"))
                gt_path.append(osp.join(data_dir, 'gt', image + ".png"))
        return image_path, gt_path


class LIPWithClass(LIP):

    def __init__(self, root, num_cls=20, train=True, transform=None, gt_transform=None):
        LIP.__init__(self, root, train, transform, gt_transform)
        self.num_cls = num_cls

    def __getitem__(self, index):
        if self.train:
            img, gt, gt_cls = self.get_a_sample(self.train_image_path, self.train_gt_path, index)
        else:
            img, gt, gt_cls = self.get_a_sample(self.val_image_path, self.val_gt_path, index)
        return img, gt, gt_cls

    def get_a_sample(self, image_path, gt_path, index):
        # get PIL Image
        # gt_cls - batch of 1D tensors of dimensionality N: N total number of classes,
        # gt_cls[i, T] = 1 if class T is present in image i, 0 otherwise
        img = Image.open(image_path[index])
        if len(img.getbands()) != 3:
            img = img.convert('RGB')
        gt = Image.open(gt_path[index])
        if len(gt.getbands()) != 1:
            gt = gt.convert('L')
        # compute gt_cls
        gt_np = np.asarray(gt, dtype=np.uint8)
        gt_cls, _ = np.histogram(gt_np, bins=self.num_cls, range=(-0.5, self.num_cls-0.5), )
        gt_cls = np.asarray(np.asarray(gt_cls, dtype=np.bool), dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return img, gt, gt_cls


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    path = 'K:\Dataset\LIP\single'

    transform_image_list = [
        transforms.Resize((512, 512), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_gt_list = [
        transforms.Resize((30, 30), Image.NEAREST),
        transforms.Lambda(lambda image: Image.fromarray(np.uint8(np.asarray(image)*(255.0/19.0)))),
        transforms.ToTensor(),
    ]

    data_transforms = {
        'image': transforms.Compose(transform_image_list),
        'gt': transforms.Compose(transform_gt_list),
    }

    loader = data.DataLoader(LIP(path, transform=data_transforms['image'], gt_transform=data_transforms['gt']),
                             batch_size=2, shuffle=False)

    for count, (src, lab) in enumerate(loader):
        src = src[0, :, :, :].numpy()
        lab = lab[0, :, :, :].numpy().transpose(1, 2, 0)


        def denormalize(image, mean, std):
            c, _, _ = image.shape
            for idx in range(c):
                image[idx, :, :] = image[idx, :, :] * std[idx] + mean[idx]
            return image

        src = denormalize(src, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).transpose(1, 2, 0)

        plt.subplot(121)
        plt.imshow(src)
        plt.subplot(122)
        plt.imshow(np.concatenate([lab, lab, lab], axis=2), cmap='gray')
        plt.show()
        print(src.shape)
        if count+1 == 4:
            break

