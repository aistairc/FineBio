# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import os, sys
import argparse
import json
import random
import time
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
from util.vis_utils import vis_detections_filtered_objects_finebio_PIL
import datasets.transforms as T

finebio_classes = np.asarray(['__background__', 'left_hand', 'right_hand', 'blue_pipette', 'yellow_pipette', 'red_pipette', '8_channel_pipette', 'blue_tip', 'yellow_tip', 'red_tip', '8_channel_tip', 'blue_tip_rack', 'yellow_tip_rack', 'red_tip_rack', '8_channel_tip_rack', '50ml_tube', '15ml_tube', 'micro_tube', '8_tube_stripes', '8_tube_stripes_lid', '50ml_tube_rack', '15ml_tube_rack', 'micro_tube_rack', '8_tube_stripes_rack', '8_tube_stripes_rack_lid', 'cell_culture_plate', 'cell_culture_plate_lid', 'trash_can', 'centrifuge', 'vortex_mixer', 'magnetic_rack', 'pcr_machine', 'tube_with_spin_column', 'spin_column', 'tube_without_lid', 'pen']) 
lhand_id = 1
rhand_id = 2

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    
    # ckpt path
    parser.add_argument("--ckpt_path", type=str, help="ckpt path")

    # dataset parameters
    parser.add_argument('--image_dir', type=str, default='data')
    parser.add_argument('--gt_path', dest='gt_path',
                        help='path to gt coco json', type=str)

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    # hand-object prediction constraints
    parser.add_argument("--use_manipulatedstate",
                      help='whether use manipulatedstate',
                      action='store_true')
    parser.add_argument("--predict_affectedobj",
                        help="whether predict affected object",
                        action='store_true')
    parser.add_argument("--use_affectedstate",
                        help="whether use affectedstate",
                        action='store_true')
    parser.add_argument("--thres_hand",
                        type=int, default=0.1)
    parser.add_argument("--thres_obj",
                        type=int, default=0.1)
    
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, _, postprocessors = build_model_main(args)
    model.to(device)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    gt_data, file2gtid = None, {}
    if args.gt_path:
        gt_data = json.load(open(args.gt_path, 'r'))
        for img in gt_data['images']:
            file2gtid[img["file_name"]] = img['id']
        gt_data = COCO(args.gt_path)
    
    imglist = os.listdir(args.image_dir)
    num_images = len(imglist)
    num_classes = len(finebio_classes)
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = T.Compose([
                    T.RandomResize([max(args.data_aug_scales)], max_size=args.data_aug_max_size),
                    normalize,
                ])

    model.eval()
    
    for i in tqdm(range(num_images)):
        im_file = os.path.join(args.image_dir, imglist[i])
        im_in = Image.open(im_file).convert("RGB")
        w, h = im_in.size
        orig_size = torch.as_tensor([int(h), int(w)]).unsqueeze(0)
        orig_size = orig_size.to(device)
        im_in, _ = transforms(im_in, None)
        im_in = im_in.to(device)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(im_in.unsqueeze(0))
        
        results = postprocessors['bbox'](outputs, orig_size)
        scores = results[0]["scores"] # (N,)
        labels = results[0]["labels"] # (N,)
        boxes = results[0]["boxes"] # (N, 4)
        hand_state = results[0]["hand_states"]  # (N, 2)
        _, hand_state_indices = torch.max(hand_state, 1)
        hand_state_indices = hand_state_indices.unsqueeze(-1).float()  # (N, 1)
        hand_vector = results[0]["hand_dxdymagnitudes"]  # (N, 3)
        manipulated_state_indices = torch.ones_like(hand_state_indices)  # (N, 1)
        affecting_state_indices = torch.ones_like(hand_state_indices)  # (N, 1)
        manip_vector = torch.zeros_like(hand_vector)  # (N, 3)
        affected_state_indices = torch.ones_like(hand_state_indices)  # (N, 1)
        if args.use_manipulatedstate:
            manipulated_state = results[0]["manipulated_states"]
            _, manipulated_state_indices = torch.max(manipulated_state, 1)
            manipulated_state_indices = manipulated_state_indices.unsqueeze(-1).float()
        if args.predict_affectedobj:
            affecting_state = results[0]["affecting_states"]
            _, affecting_state_indices = torch.max(affecting_state, 1)
            affecting_state_indices = affecting_state_indices.unsqueeze(-1).float()
            manip_vector = results[0]["manip_dxdymagnitudes"]
        if args.use_affectedstate:
            affected_state = results[0]["affected_states"]
            _, affected_state_indices = torch.max(affected_state, 1)
            affected_state_indices = affected_state_indices.unsqueeze(-1).float()
                
        dets = {}
        for j in range(1, num_classes):
            inds = torch.nonzero(labels == j).view(-1)
            if inds.numel() > 0:
                thres_score = args.thres_obj
                if j == lhand_id or j == rhand_id:
                    thres_score = args.thres_hand
                inds = inds[scores[inds] > thres_score]
                cls_scores = scores[inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = boxes[inds]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), hand_state_indices[inds], hand_vector[inds], manipulated_state_indices[inds], affecting_state_indices[inds], affected_state_indices[inds]), 1)
                if j != lhand_id and j != rhand_id:
                    cls_dets[:, 6:9] = manip_vector[inds]
                cls_dets = cls_dets[order]
                dets[finebio_classes[j]] = cls_dets.cpu().numpy()
        
        # visualization
        im2show = cv2.imread(im_file)
        im2show = vis_detections_filtered_objects_finebio_PIL(im2show, dets, args.thres_hand, args.thres_obj, predict_affectedobj=args.predict_affectedobj)
        if gt_data:
            frame = Image.new(im2show.mode, (im2show.size[0], im2show.size[1] + 165), (255, 255, 255))
            frame.paste(im2show, (0, 0))
            draw = ImageDraw.Draw(frame)
            img_id = file2gtid[imglist[i]]
            ann_ids = gt_data.getAnnIds(img_id)
            manip_l, manip_r, affected_l, affected_r = [], [], [], []
            for ann_id in ann_ids:
                ann = gt_data.loadAnns(ann_id)[0]
                if "manipulated_left" in ann["attributes"]["object_state"]:
                    manip_l.append(gt_data.loadCats(ann["category_id"])[0]["name"])
                if "manipulated_right" in ann["attributes"]["object_state"]:
                    manip_r.append(gt_data.loadCats(ann["category_id"])[0]["name"])
                if "affected_left" in ann["attributes"]["object_state"]:
                    affected_l.append(gt_data.loadCats(ann["category_id"])[0]["name"])
                if "affected_right" in ann["attributes"]["object_state"]:
                    affected_r.append(gt_data.loadCats(ann["category_id"])[0]["name"])
            draw.text((5, im2show.size[1]), "Ground Truth",font=ImageFont.truetype('util/times_b.ttf', size=50), fill=(0,0,0))
            draw.text((5, im2show.size[1] + 55), f"manipulated-left: {','.join(manip_l) if len(manip_l) else 'none'} / affected-left: {','.join(affected_l) if len(affected_l) else 'none'}",font=ImageFont.truetype('util/times_b.ttf', size=50), fill=(0,0,255))
            draw.text((5, im2show.size[1] + 110), f"manipulated-right: {','.join(manip_r) if len(manip_r) else 'none'} / affected-right: {','.join(affected_r) if len(affected_r) else 'none'}", font=ImageFont.truetype('util/times_b.ttf', size=50), fill=(255,0,0))
            im2show = frame.copy() 
            
        folder_name = args.output_dir
        os.makedirs(folder_name, exist_ok=True)
        result_path = os.path.join(folder_name, imglist[i][:-4] + "_det.png")
        im2show.save(result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
