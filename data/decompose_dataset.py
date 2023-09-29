import glob
import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class DecomposeDataset(data.Dataset):

    def __init__(self, opt):
        self._video_dir = opt['video_dir']
        self.downsample_factor = opt['downsample_factor']

        self._video_names = []
        self._frame_nums = []

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ])

        self.xflip = opt['xflip']

        video_names_list = open(opt['data_name_txt'], 'r').readlines()

        for row in video_names_list:
            video_name = row.split()[0]
            frame_num = int(row.split()[1])
            self._video_names.append(video_name)
            self._frame_nums.append(frame_num)

    def _load_raw_image(self, img_path):
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image.load()
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)

        return image

    def __getitem__(self, index):
        video_name = self._video_names[index]

        random_frame_idx = random.randint(30, self._frame_nums[index] - 30)

        img_path = f'{self._video_dir}/{video_name}/{random_frame_idx:03d}.png'
        random_frame = self._load_raw_image(img_path)

        random_frame_aug = self.transform(random_frame)

        random_frame = np.array(random_frame).transpose(2, 0,
                                                        1).astype(np.float32)
        random_frame_aug = np.array(random_frame_aug).transpose(
            2, 0, 1).astype(np.float32)

        identity_image = self._load_raw_image(
            f'{self._video_dir}/{video_name}/000.png')
        identity_image = np.array(identity_image).transpose(2, 0, 1).astype(
            np.float32)

        if self.xflip and random.random() > 0.5:
            random_frame = random_frame[:, :, ::-1].copy()  # [C, H ,W]
            random_frame_aug = random_frame_aug[:, :, ::-1].copy()

        random_frame = random_frame / 127.5 - 1
        random_frame_aug = random_frame_aug / 127.5 - 1
        identity_image = identity_image / 127.5 - 1

        random_frame = torch.from_numpy(random_frame)
        identity_image = torch.from_numpy(identity_image)

        return_dict = {
            # 'densepose': pose,
            'video_name': f'{video_name}_{random_frame_idx:03d}',
            'frame_img': random_frame,
            'frame_img_aug': random_frame_aug,
            'identity_image': identity_image
        }

        return return_dict

    def __len__(self):
        return len(self._video_names)


class DecomposeMixDataset(data.Dataset):

    def __init__(self, opt):
        self._video_dir = opt['video_dir']
        self.downsample_factor = opt['downsample_factor']

        _video_names = []
        _frame_nums = []

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ])

        self.translate_transform = transforms.Compose(
            [transforms.RandomAffine(degrees=0, translate=(0.1, 0.3))])

        self.xflip = opt['xflip']

        video_names_list = open(opt['data_name_txt'], 'r').readlines()

        for row in video_names_list:
            video_name = row.split()[0]
            frame_num = int(row.split()[1])
            _video_names.append(video_name)
            _frame_nums.append(frame_num)

        self._shhq_data_dir = opt['shhq_data_dir']
        self._ann_dir = opt['ann_dir']

        _image_fnames = []
        # for idx, row in enumerate(
        #         open(os.path.join(f'{self._ann_dir}/upper_fused.txt'), 'r')):
        #     annotations = row.split()
        #     if len(annotations[:-1]) == 1:
        #         img_name = annotations[0]
        #     else:
        #         img_name = ''
        #         for name in annotations[:-1]:
        #             img_name += f'{name}\xa0'
        #         img_name = img_name[:-1]
        #     _image_fnames.append(img_name)
        shhq_path_list = glob.glob(f'{self._shhq_data_dir}/*.png')
        for shhq_path in shhq_path_list:
            _image_fnames.append(shhq_path.split('/')[-1])

        augment_times = max(1, len(_image_fnames) // len(_video_names))

        augmented_videos = _video_names * augment_times
        self._frame_nums = _frame_nums * augment_times
        self._all_file_name = augmented_videos + _image_fnames
        self.video_num = len(augmented_videos)

    def _load_raw_image(self, img_path):
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            image.load()
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)

        return image

    def sample_video_data(self, index):
        video_name = self._all_file_name[index]

        random_frame_idx = random.randint(30, self._frame_nums[index] - 30)

        img_path = f'{self._video_dir}/{video_name}/{random_frame_idx:03d}.png'
        random_frame = self._load_raw_image(img_path)

        random_frame_aug = self.transform(random_frame)

        random_frame = np.array(random_frame).transpose(2, 0,
                                                        1).astype(np.float32)
        random_frame_aug = np.array(random_frame_aug).transpose(
            2, 0, 1).astype(np.float32)

        identity_image = self._load_raw_image(
            f'{self._video_dir}/{video_name}/000.png')
        identity_image = np.array(identity_image).transpose(2, 0, 1).astype(
            np.float32)

        if self.xflip and random.random() > 0.5:
            random_frame = random_frame[:, :, ::-1].copy()  # [C, H ,W]
            random_frame_aug = random_frame_aug[:, :, ::-1].copy()

        return identity_image, random_frame, random_frame_aug

    def sample_img_data(self, index):
        img_name = self._all_file_name[index]

        img_path = f'{self._shhq_data_dir}/{img_name}'

        identity_image = self._load_raw_image(img_path)

        random_frame = self.translate_transform(identity_image)

        random_frame_aug = self.transform(random_frame)

        random_frame = np.array(random_frame).transpose(2, 0,
                                                        1).astype(np.float32)
        random_frame_aug = np.array(random_frame_aug).transpose(
            2, 0, 1).astype(np.float32)
        identity_image = np.array(identity_image).transpose(2, 0, 1).astype(
            np.float32)

        if self.xflip and random.random() > 0.5:
            random_frame = random_frame[:, :, ::-1].copy()  # [C, H ,W]
            random_frame_aug = random_frame_aug[:, :, ::-1].copy()

        return identity_image, random_frame, random_frame_aug

    def __getitem__(self, index):

        if index < self.video_num:
            identity_image, random_frame, random_frame_aug = self.sample_video_data(
                index)
        else:
            identity_image, random_frame, random_frame_aug = self.sample_img_data(
                index)

        random_frame = random_frame / 127.5 - 1
        random_frame_aug = random_frame_aug / 127.5 - 1
        identity_image = identity_image / 127.5 - 1

        random_frame = torch.from_numpy(random_frame)
        random_frame_aug = torch.from_numpy(random_frame_aug)
        identity_image = torch.from_numpy(identity_image)

        return_dict = {
            'frame_img': random_frame,
            'frame_img_aug': random_frame_aug,
            'identity_image': identity_image
        }

        return return_dict

    def __len__(self):
        return len(self._all_file_name)
