import torch, os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import transforms
from PIL import Image

class VOCSegmentation(Dataset):
    def __init__(self, img_path, gt_path, txt_file, train_val = "train", base_size=520, crop_size=480, filp_prob=0.5):
        super().__init__()

        if train_val == 'train' :
            with open(txt_file, 'r') as tf:
                img_name_list = [name.strip() for name in tf.readlines()]
            self.transforms = transforms.Compose([transforms.RandomResize(int(base_size*0.5), int(base_size*2)),
                                                  transforms.RandomHorizonFlip(filp_prob),
                                                  transforms.RandomCrop(crop_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                  ])
        else:
            with open(txt_file, 'r') as tf:
                img_name_list = [name.strip() for name in tf.readlines()]
            self.transforms = transforms.Compose([transforms.RandomResize(base_size, base_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                  ])

        self.img_files = [os.path.join(img_path, name + '.jpg') for name in img_name_list]
        self.gt_files = [os.path.join(gt_path, name + '.png') for name in img_name_list]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image = Image.open(self.img_files[index])
        target = Image.open(self.gt_files[index])
        img, target = self.transforms(image, target)
        return img, target

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, pad=0)
        batched_targets = cat_list(targets, pad=255)
        return batched_imgs, batched_targets

def cat_list(images, pad=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(pad)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

