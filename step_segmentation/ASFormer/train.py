import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random
from glob import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--split', default='')
parser.add_argument('--run_dir', type=str)
parser.add_argument('--test_subset', type=str, default="valid")

args = parser.parse_args()
 
num_epochs = 120

lr = 0.0005
num_layers = 11
num_decoders = 5
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1

print(f"#decoders: {num_decoders}, #layers: {num_layers}")

if len(args.split):
    vid_list_file = args.run_dir + "/data/splits/train_split" + args.split + ".bundle"
    vid_list_file_tst = args.run_dir + f"/data/splits/{args.test_subset}_split" + args.split + ".bundle"
    mapping_file = args.run_dir + "/data/mapping_" + args.split + ".txt"
    model_dir = args.run_dir + f"/ASFormer/models/split_" + args.split + f"_{num_decoders}d_layer{num_layers}"
else:
    vid_list_file = args.run_dir + "/data/splits/train.bundle"
    vid_list_file_tst = args.run_dir + f"/data/splits/{args.test_subset}.bundle"
    mapping_file = args.run_dir + "/data/mapping.txt"
    model_dir = args.run_dir + f"/ASFormer/models/all" + f"_{num_decoders}d_layer{num_layers}"
features_path = args.run_dir + f"/data/rgbflow_i3d_features/"
gt_path = args.run_dir + "/data/groundTruth/"
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_decoders, num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, "finebio", args.split, run_dir=args.run_dir)

os.makedirs(model_dir, exist_ok=True)
batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
batch_gen.read_data(vid_list_file)

batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
batch_gen_tst.read_data(vid_list_file_tst)

trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)
