from util.transform import *
from copy import deepcopy
import math
import numpy as np
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):     
    def __init__(self, name, root, mode, id_path, cover_radio=None, block_size=None, nsample=None, crop_size=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = crop_size
        self.cover_radio = cover_radio
        self.block_size = block_size

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()
        if nsample is not None:
            self.ids *= math.ceil(nsample / len(self.ids))
            self.ids = self.ids[:nsample]

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s, img_m = deepcopy(img), deepcopy(img), deepcopy(img) 

        img_s, cutmix_box = self.mix_image_transformations(img_s)
        img_m, cover_mask = self.mix_image_transformations(img_m)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()

        ignore_mask[mask == 254] = 255
        
        img_s = normalize(img_s, ignore_mask=ignore_mask)
        img_m = normalize(img_m, ignore_mask=ignore_mask)

        return normalize(img_w), img_s, img_m, ignore_mask, cutmix_box, cover_mask
        
    def mix_image_transformations(self, img):
        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = blur(img, p=0.5)
        cutmix_box = obtain_cutmix_box(img.size[0], p=1/2)
        return img, cutmix_box
    
    def mask_image_transformations(self, img):
        
        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = blur(img, p=0.5)
        cover_mask = torch.zeros(img.size[0], img.size[0])

        return img, cover_mask

        # choice = random.randint(1, 2)   # Trusted Mask Perturbation

        # if choice == 1:
        #     if random.random() < 0.8:
        #         img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(img)
        #     img = blur(img, p=0.5)
        #     cover_mask = block_mask(img.size[0], self.cover_radio, self.block_size)
        #     return img, cover_mask
        # else:
        #     if random.random() < 0.8:
        #         img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        #     img = transforms.RandomGrayscale(p=0.2)(img)
        #     img = blur(img, p=0.5)
        #     cover_mask = torch.zeros(img.size[0], img.size[0])
        # return img, cover_mask
    
    def __len__(self):
        return len(self.ids)

