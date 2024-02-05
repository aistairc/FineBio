#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
from glob import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)

parser.add_argument('--run_dir', type=str)

parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1

if len(args.split):
    vid_list_file = args.run_dir + "/data/splits/train_split" + args.split + ".bundle"
    mapping_file = args.run_dir + "/data/mapping_" + args.split + ".txt"
    model_dir = args.run_dir + f"/MS-TCN2/models/split_" + args.split + "_lr{:.0e}_PG{}_R{}*{}".format(lr, num_layers_PG, num_layers_R, num_R)
else:
    vid_list_file = args.run_dir + "/data/splits/train.bundle"
    mapping_file = args.run_dir + "/data/mapping.txt"
    model_dir = args.run_dir + f"/MS-TCN2/models/all" + args.split + "_lr{:.0e}_PG{}_R{}*{}".format(lr, num_layers_PG, num_layers_R, num_R)
features_path = args.run_dir + f"/data/rgbflow_i3d_features/"
gt_path = args.run_dir + "/data/groundTruth/"

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, "finebio", run_dir=args.run_dir)
os.makedirs(model_dir, exist_ok=True)
batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
batch_gen.read_data(vid_list_file)
trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
