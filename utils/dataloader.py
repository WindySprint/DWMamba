import os
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2

from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

def train_val_list(enhan_images_path, ori_images_path):
    image_list_index = os.listdir(ori_images_path) 
    all_length = len(image_list_index)
    image_list_index = random.sample(image_list_index, all_length)

    image_dataset = []
    for i in image_list_index:  # Add paths and combine them
        image_dataset.append((enhan_images_path + i, ori_images_path + i))

    train_list = image_dataset[:int(all_length*0.9)]
    val_list = image_dataset[int(all_length*0.9):]

    return train_list, val_list

class train_val_loader(data.Dataset):

    def __init__(self, enhan_images_path, ori_images_path, mode='train'):

        self.train_list, self.val_list = train_val_list(enhan_images_path, ori_images_path)
        self.mode = mode
        if self.mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_ori_path = self.data_list[index]

        data_clean = Image.open(data_clean_path)
        data_ori = Image.open(data_ori_path)
        data_canny = cv2.imread(data_ori_path, cv2.IMREAD_GRAYSCALE)
        data_canny = cv2.Canny(data_canny, 50, 150)

        data_clean = np.asarray(data_clean) / 255.0
        data_ori = np.asarray(data_ori) / 255.0
        data_canny = np.asarray(data_canny) / 255.0

        data_clean = torch.from_numpy(data_clean).float()
        data_ori = torch.from_numpy(data_ori).float()
        data_canny = torch.from_numpy(data_canny).unsqueeze(2).float()

        return data_clean.permute(2, 0, 1), data_ori.permute(2, 0, 1), data_canny.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

class test_loader(data.Dataset):
    def __init__(self, ori_images_path):
        super(test_loader, self).__init__()

        image_list_index = sorted(os.listdir(ori_images_path))
        self.image_dataset = [os.path.join(ori_images_path, x) for x in image_list_index if is_image_file(x)]
        self.all_length = len(self.image_dataset)

    def __len__(self):
        return self.all_length

    def __getitem__(self, index):

        data_ori_path = self.image_dataset[index]
        filename = data_ori_path.split('/')[-1]
        data_ori = Image.open(data_ori_path)
        data_canny = cv2.imread(data_ori_path, cv2.IMREAD_GRAYSCALE)
        data_canny = cv2.Canny(data_canny, 50, 150)

        data_ori = np.asarray(data_ori) / 255.0
        data_canny = np.asarray(data_canny) / 255.0

        data_ori = torch.from_numpy(data_ori).float()
        data_canny = torch.from_numpy(data_canny).unsqueeze(2).float()

        return data_ori.permute(2, 0, 1), data_canny.permute(2, 0, 1), filename