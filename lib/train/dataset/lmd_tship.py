import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class LMD_TShip(BaseVideoDataset):
    """ LaSOT dataset.

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_names=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lmd_tship_dir if root is None else root
        super().__init__('LMD_TShip',root, image_loader)
        
        
        # seq_names is the index of the folder inside the got10k root path
        if split is not None:
            if seq_names is not None:
                raise ValueError('Cannot set both split_name and seq_names.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train_full':
                file_path = os.path.join(ltr_path, 'data_specs', 'lmd_tship_full.txt')
            elif split == 'vottrain':
                file_path = os.path.join(ltr_path, 'data_specs', 'lmd_tship_vot_train_split.txt')
            elif split == 'vottest':
                file_path = os.path.join(ltr_path, 'data_specs', 'lmd_tship_vot_test_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_names = np.loadtxt(file_path, dtype=str)
        
        self.sequence_list = seq_names

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()
        
    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'Protocol I', 'All Sequences.txt')) as f:
            dir_list = np.loadtxt(f, dtype=str)
        
        return dir_list
   
    def _build_class_list(self):
        seq_per_class = {}
        class_file_dir = os.path.join(self.root, 'Protocol I', 'Ship Category')
        class_files = os.listdir(class_file_dir)
        for class_file in class_files:
            seq_list = np.loadtxt(os.path.join(class_file_dir, class_file), dtype=str)
            for seq_name in seq_list:
                class_name = class_file.split('/')[-1].split('.')[0]
                if class_name in seq_per_class:
                    seq_per_class[class_name].append(seq_name)
                else:
                    seq_per_class[class_name] = [seq_name]
                    
        return seq_per_class
    
    def get_name(self):
        return 'LMD-TShip'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = np.loadtxt(bb_anno_file, delimiter=',', dtype=np.float32)
        return torch.tensor(gt)

    # def _read_target_visible(self, seq_path):
    #     # Read full occlusion and out_of_view
    #     occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
    #     out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

    #     with open(occlusion_file, 'r', newline='') as f:
    #         occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
    #     with open(out_of_view_file, 'r') as f:
    #         out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

    #     target_visible = ~occlusion & ~out_of_view

    #     return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, 'data', self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_id):
        for seq_class, seq_list in self.seq_per_class.items():
            if self.sequence_list[seq_id] in seq_list:
                raw_class = seq_class
                continue
                
        return raw_class
    
    def get_class_name(self, seq_id):
        seq_name = self._get_sequence_path(seq_id).split('/')[-1]
        obj_class = self._get_class(seq_name)

        return obj_class

    def get_frames(self, seq_name, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_name)

        obj_class = self._get_class(seq_name)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_name)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
