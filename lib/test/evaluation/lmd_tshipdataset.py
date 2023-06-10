import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LMD_TShipDataset(BaseDataset):
    """
    LMD_TShip test set consisting of 39 videos (see Protocol-II in the LMD_TShip paper)

    Publication:
        LMD-TShip⋆: Vision Based Large-Scale Maritime Ship Tracking Benchmark for Autonomous Navigation Applications
        Shan, Yunxiao and Liu, Shanghua and Zhang, Yunfei and Jing, Min and Xu, Huawei
        IEEE Access, 2021
        https://arxiv.org/pdf/1809.07845.pdf

    See the dataset from https://yat-sen-robot.github.io/usilab-web/#/evaluation
    """
    def __init__(self, split):
        super().__init__()
        self.base_path = self.env_settings.lmd_tship_path
        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/data/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/data/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'lmd_tship', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        if split == 'protocol1':
            with open('{}/lmd_tship_full.txt'.format(self.env_settings.dataspec_path)) as f:
                sequence_list = f.read().splitlines()
                
        elif split == 'protocol2':
            with open('{}/lmd_tship_vot_test_split.txt'.format(self.env_settings.dataspec_path)) as f:
                sequence_list = f.read().splitlines()

        return sequence_list
