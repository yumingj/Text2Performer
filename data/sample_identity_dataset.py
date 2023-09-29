import json
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class SampleIdentityDataset(data.Dataset):

    def __init__(self, opt):
        self._video_dir = opt['video_dir']
        self.downsample_factor = opt['downsample_factor']

        self._video_names = []
        self._frame_nums = []

        self.xflip = opt['xflip']

        video_names_list = open(opt['data_name_txt'], 'r').readlines()

        for row in video_names_list:
            video_name = row.split()[0]
            self._video_names.append(video_name)

        with open(opt['text_file']) as json_file:
            self.text_descriptions = json.load(json_file)

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

        identity_image = self._load_raw_image(
            f'{self._video_dir}/{video_name}/000.png')
        identity_image = np.array(identity_image).transpose(2, 0, 1).astype(
            np.float32)
        if self.xflip and random.random() > 0.5:
            identity_image = identity_image[:, :, ::-1].copy()  # [C, H ,W]

        identity_image = identity_image / 127.5 - 1

        identity_image = torch.from_numpy(identity_image)

        if len(self.text_descriptions[video_name]) == 1:
            text_description = self.text_descriptions[video_name]
        else:
            text_description = random.choice(
                self.text_descriptions[video_name])

        return_dict = {
            'img_name': video_name,
            'image': identity_image,
            'text': text_description
        }

        return return_dict

    def __len__(self):
        return len(self._video_names)
