import os
import json
import numpy as np
import pandas as pd
from glob import glob
import re
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
    

@register_dataset("finebio")
class FineBioDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json files for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim  
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        type_names,       # list of types used in annotation  (verb, manipulated, affected, atomic_operation)
        hands,            # list of hand of which action is detected. if empty, the detection is not hand-wise. ([left], [right], [left, right], [])
    ):
        # file path
        assert os.path.exists(feat_folder)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

        # list of hand to include
        self.hands = hands
        if len(self.hands) == 0:
            self.hands = [None]
            
        assert len(type_names), "no type specified"
        assert ("hand" not in type_names) or (len(hands) == 0), "if hand head exists, other type prediction won't be hand-wise."

        # load database and select the subset
        self.data_lists = []
        self.label_dicts = []
        self.num_classes = []
        self.original_type_names = []
        self.type_names = []
        self.type_groups_with_same_hand = []  # lists of type indexes for (verb, manipulated, affected). if length of hands is multiple, create different list for each hand.

        idx = 0
        for hand in self.hands:
            type_group = []
            for type_name in type_names:
                label_dict = self._load_label_dict(type_name)
                self.label_dicts.append(label_dict)
                dict_db, num_class = self._load_json_db(self.json_file, type_name, hand, label_dict)
                self.data_lists.append(dict_db)
                self.num_classes.append(num_class)
                self.original_type_names.append(type_name)
                new_type_name = type_name if hand is None else f"{hand}_{type_name}"
                self.type_names.append(new_type_name)
                type_group.append(idx)
                idx += 1
            self.type_groups_with_same_hand.append(type_group)

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'finebio',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes 
    
    def _load_label_dict(self, type_name):
        file_name = ""
        if type_name == "verb":
            file_name = "data/annotations/verb_to_int.json"
        elif type_name in ["manipulated", "affected"]:
            file_name = "data/annotations/object_to_int.json"
        elif type_name == "atomic_operation":
            file_name = "data/annotations/atomic_operation_to_int.json"
        with open(file_name, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def _load_json_db(self, json_file, type_name, hand, label_dict):
        assert hand in [None, "left", "right"], "hand should be either none or left or right"
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        num_classes = json_data["type_info"][type_name]["num_classes"]

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                segments, labels = [], []
                for act in value['annotations']:
                    # no hand specification or
                    # hand is specified and the segment is done by the hand
                    if hand is None or act["hand_label"] == hand:
                        segments.append(act['segment'])
                        labels.append([label_dict[act[f'{type_name}_label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)

            else:
                segments = None
                labels = None
            
            dict_db += ({
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels
            }, )

        return dict_db, num_classes

    def __len__(self):
        return len(self.data_lists[0])

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_items = [data_list[idx] for data_list in self.data_lists]

        # load features
        filename = os.path.join(self.feat_folder,
                                self.file_prefix + video_items[0]['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_items[0]['segments'] is not None:
            # if multiple types are predicted, segments should be all the same while labels are different.
            segments = torch.from_numpy(
                video_items[0]['segments'] * video_items[0]['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_items[0]['labels']) if len(video_items) == 1 \
                else [torch.from_numpy(video_items[i]['labels']) for i in range(len(video_items))]
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_items[0]['id'],
                    'feats'           : feats,      # C x T
                    'segments'        : segments,   # N x 2
                    'labels'          : labels,     # N × #(type_names)
                    'fps'             : video_items[0]['fps'],
                    'duration'        : video_items[0]['duration'],
                    'feat_stride'     : feat_stride,
                    'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        # if two hand annotations exist, return list
        return data_dict
