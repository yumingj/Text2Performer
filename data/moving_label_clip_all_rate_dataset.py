import json
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def proper_capitalize(text):
    if len(text) > 0:
        text = text.lower()
        text = text[0].capitalize() + text[1:]
        for idx, char in enumerate(text):
            if char in ['.', '!', '?'] and (idx + 2) < len(text):
                text = text[:idx + 2] + text[idx + 2].capitalize() + text[idx +
                                                                          3:]
        text = text.replace(' i ', ' I ')
        text = text.replace(',i ', ',I ')
        text = text.replace('.i ', '.I ')
        text = text.replace('!i ', '!I ')
    return text


class MovingLabelsClipAllRateTextDataset(data.Dataset):

    def __init__(self, opt):
        self._video_dir = opt['video_dir']
        self.downsample_factor = opt['downsample_factor']
        self.random_start = opt['random_start']
        self._video_names = []

        self.frame_sample_rate = opt['frame_sample_rate']

        self._action_label_folder = opt['action_label_folder']
        self.action_labels = []

        self.fixed_video_len = opt['fixed_video_len']
        self.xflip = opt['xflip']

        video_names_list = open(opt['data_name_txt'], 'r').readlines()

        self.moving_dict = np.load(
            opt['moving_frame_dict'], allow_pickle=True).item()

        self.interpolation_rate = opt['interpolation_rate']

        self.all_clip_start_frame_list = []
        self.all_clip_end_frame_list = []
        self.all_clip_action_label_list = []
        self.frame_num_list = []

        with open(opt['overall_caption_templates_file'], 'r') as f:
            self.overall_caption_templates = json.load(f)

        for row in video_names_list:
            video_name = row.split()[0]
            frame_nums = int(row.split()[1])
            action_label_txt = open(
                f'{self._action_label_folder}/{video_name}.txt',
                'r').readlines()

            clip_start_frame_list = []
            clip_end_frame_list = []
            clip_action_label_list = []
            for action_row in action_label_txt:
                start_frame, end_frame, action_label = action_row[:-1].split()
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                action_label = int(action_label)

                if (end_frame - start_frame
                    ) < self.frame_sample_rate * self.fixed_video_len:
                    continue

                clip_start_frame_list.append(start_frame)
                clip_end_frame_list.append(end_frame)
                clip_action_label_list.append(action_label)

            if len(clip_start_frame_list) == 0:
                continue

            self._video_names.append(video_name)
            self.all_clip_start_frame_list.append(clip_start_frame_list)
            self.all_clip_end_frame_list.append(clip_end_frame_list)
            self.all_clip_action_label_list.append(clip_action_label_list)
            self.frame_num_list.append(frame_nums)

        assert len(self._video_names) == len(self.all_clip_start_frame_list)
        assert len(self._video_names) == len(self.all_clip_end_frame_list)
        assert len(self._video_names) == len(self.all_clip_action_label_list)
        assert len(self._video_names) == len(self.frame_num_list)

    def _load_raw_image(self, img_path):
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.LANCZOS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image.astype(np.float32)

    def sample_motion_clip(self, index):
        clip_start_frame_list = self.all_clip_start_frame_list[index]
        clip_end_frame_list = self.all_clip_end_frame_list[index]
        clip_action_label_list = self.all_clip_action_label_list[index]

        num_clip = len(clip_start_frame_list)

        clip_index = random.randint(0, num_clip - 1)

        action_label_list = clip_action_label_list[clip_index]

        clip_idx = list(
            range(clip_start_frame_list[clip_index],
                  clip_end_frame_list[clip_index] + 1))

        segm = len(clip_idx) // self.fixed_video_len

        segm_dist = []
        for i in range(self.fixed_video_len - 1):
            segm_dist.append(segm)

        for i in range(
                min(
                    len(clip_idx) - sum(segm_dist) - 2,
                    self.fixed_video_len - 1)):
            segm_dist[i] += 1

        frame_idx_list = []
        frame_idx_list.append(clip_start_frame_list[clip_index])

        for i in range(len(segm_dist) - 1):
            frame_idx_list.append(clip_start_frame_list[clip_index] +
                                  sum(segm_dist[:i + 1]))
        frame_idx_list.append(clip_end_frame_list[clip_index])

        return frame_idx_list, action_label_list

    def sample_random_clip(self, index):

        video_name = self._video_names[index]

        video_len = self.fixed_video_len

        if len(self.moving_dict[video_name]) == 0:
            start_frame = random.randint(
                30, self.frame_num_list[index] - 1 - video_len)
        else:
            start_frame = random.choice(self.moving_dict[video_name])

            while ((start_frame + video_len) >
                   (self.frame_num_list[index] - 1)):
                start_frame = random.choice(self.moving_dict[video_name])

        frame_idx_list = []
        for frame_idx in range(video_len):
            video_frame_idx = start_frame + frame_idx
            frame_idx_list.append(video_frame_idx)

        action_label_list = 22

        return frame_idx_list, action_label_list

    def generate_caption(self, label):
        caption = random.choice(self.overall_caption_templates[label])
        replacing_word = random.choice(
            self.overall_caption_templates["gender"])
        caption = caption.replace('<Gender>', replacing_word)
        caption = proper_capitalize(caption)

        return caption

    def __getitem__(self, index):
        video_name = self._video_names[index]

        if np.random.uniform(low=0.0, high=1.0) < self.interpolation_rate:
            frame_idx_list, action_label_list = self.sample_random_clip(index)
            interpolation_mode = True
        else:
            frame_idx_list, action_label_list = self.sample_motion_clip(index)
            interpolation_mode = False

        frames = []
        for frame_idx in frame_idx_list:
            img_path = f'{self._video_dir}/{video_name}/{frame_idx:03d}.png'
            frames.append(self._load_raw_image(img_path))

        frames = np.stack(frames, axis=0)

        exemplar_img = self._load_raw_image(
            f'{self._video_dir}/{video_name}/000.png')

        frames = frames / 127.5 - 1
        exemplar_img = exemplar_img / 127.5 - 1

        frames = torch.from_numpy(frames)
        exemplar_img = torch.from_numpy(exemplar_img)

        if action_label_list == 22:
            text_description = 'empty'
        else:
            text_description = self.generate_caption(str(action_label_list))

        return_dict = {
            'video_name': video_name,
            'video_frames': frames,
            'video_len': self.fixed_video_len,
            'exemplar_img': exemplar_img,
            'action_labels': action_label_list,
            'text': text_description,
            'interpolation_mode': interpolation_mode
        }

        return return_dict

    def __len__(self):
        return len(self._video_names)
