import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from libs.core import load_config
from libs.datasets import make_dataset
from libs.utils import ANETdetection, convert_pkl2json

# you can custom here
parser = argparse.ArgumentParser()
parser.add_argument("--op_pkl_file", type=str, help="path for atomic operation detection pkl file")
parser.add_argument("--config", type=str, help="path for config", default="configs/finebio_i3d_atomic_operation.yaml")
parser.add_argument("--stat_dir", type=str, help="path for statistics directory", default="data/statistics")
parser.add_argument("--op2int_file", type=str, help="path for atomic operation label dictionary", default="data/annotations/atomic_operation_to_int.json")
parser.add_argument("--verb2int_file", type=str, help="path for verb label dictionary", default="data/annotations/verb_to_int.json")
parser.add_argument("--obj2int_file", type=str, help="path for object label dictionary", default="data/annotations/object_to_int.json")
hand_annotation = []  # TODO: implement for [left, right]
args = parser.parse_args()

action_pred_json_file = args.op_pkl_file.replace('.pkl', '.json')
if not os.path.exists(action_pred_json_file):
    convert_pkl2json(args.op_pkl_file, args.op2int_file, output_path=action_pred_json_file)

# save_dir = "/work/ohashi/AIP/scripts/action_detection/tmp"
save_dir = '/'.join(action_pred_json_file.split('/')[:-1])
if os.path.exists(os.path.join(save_dir, "results.txt")):
    os.remove(os.path.join(save_dir, "results.txt"))

atomop2int = json.load(open(args.op2int_file, 'r'))
verb2int = json.load(open(args.verb2int_file, 'r'))
obj2int = json.load(open(args.obj2int_file, 'r'))

label2int_dict = {"atomic_operation": atomop2int, "verb": verb2int, "manipulated": obj2int, "affected": obj2int}


def split_label(atomop_label, type="atomic_operation"):
    if type == "atomic_operation":
        return atomop_label
    elif type == "verb":
        return atomop_label.split('-')[0]
    elif type == "manipulated":
        return atomop_label.split('-')[1]
    else:
        return atomop_label.split('-')[2] if len(atomop_label.split('-')) == 3 else "none"


def load_pred_seg_from_json(json_file, type="atomic_operation"):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['results']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            label_id = label2int_dict[type][split_label(event['label'], type)]
            labels += [label_id]
            scores += [float(event['score'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id': vids,
        't-start': starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


def save_json(pred, save_path, type="atomic_operation"):
    new_pred = {"results": defaultdict(list)}
    label2int = label2int_dict[type]
    int2label = dict(zip(label2int.values(), label2int.keys()))
    for i in range(len(pred)):
        p = pred.iloc[i]
        video_id = p["video-id"]
        new_pred["results"][video_id].append(
            {
                "score": p["score"],
                "segment": [p["t-start"], p["t-end"]],
                "label": int2label[p["label"]]
            }
        )
    with open(save_path, 'w') as f:
        json.dump(new_pred, f)


cfg = load_config(args.config)
cfg['dataset']['type_names'] = ["verb", "manipulated", "affected", "atomic_operation"]
cfg['dataset']['hands'] = hand_annotation
val_dataset = make_dataset(
    cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
)
action_eval = ANETdetection(
    val_dataset.data_lists[3],
    val_dataset.split[0],
    tiou_thresholds=np.linspace(0.3, 0.7, 5),
    dataset_name="atomic_operation",
    label_dict=atomop2int,
    stat_dir=args.stat_dir
)
action_pred = load_pred_seg_from_json(action_pred_json_file, "atomic_operation")
mAP, avg_mAP, mRecall = action_eval.evaluate(action_pred)
with open(os.path.join(save_dir, "results.txt"), 'a') as f:
    block = '[RESULTS] Action detection results on {:s}.'.format(action_eval.dataset_name)
    for tiou, tiou_mAP, tiou_mRecall in zip(action_eval.tiou_thresholds, mAP, mRecall):
        block += '\n|tIoU = {:.2f}: '.format(tiou)
        block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
        for idx, k in enumerate(action_eval.top_k):
            block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
    block += '\nAverage mAP: {:>4.2f} (%)\n'.format(avg_mAP*100)
    f.write(block)
action_eval.plot_ap(save_dir)
action_eval.plot_recall(save_dir)
action_eval.get_confusion_matrix(save_dir)

verb_eval = ANETdetection(
    val_dataset.data_lists[0],
    val_dataset.split[0],
    tiou_thresholds=np.linspace(0.3, 0.7, 5),
    dataset_name="verb",
    label_dict=verb2int,
    stat_dir=args.stat_dir
)
verb_pred = load_pred_seg_from_json(action_pred_json_file, "verb")
save_json(verb_pred, action_pred_json_file.replace("atomic_operation_eval_results.json", "verb_eval_results.json"), "verb")
mAP, avg_mAP, mRecall = verb_eval.evaluate(verb_pred)
with open(os.path.join(save_dir, "results.txt"), 'a') as f:
    block = '[RESULTS] Action detection results on {:s}.'.format(verb_eval.dataset_name)
    for tiou, tiou_mAP, tiou_mRecall in zip(verb_eval.tiou_thresholds, mAP, mRecall):
        block += '\n|tIoU = {:.2f}: '.format(tiou)
        block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
        for idx, k in enumerate(verb_eval.top_k):
            block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
    block += '\nAverage mAP: {:>4.2f} (%)\n'.format(avg_mAP*100)
    f.write(block)
verb_eval.plot_ap(save_dir)
verb_eval.plot_recall(save_dir)
verb_eval.get_confusion_matrix(save_dir)

manip_eval = ANETdetection(
    val_dataset.data_lists[1],
    val_dataset.split[0],
    tiou_thresholds=np.linspace(0.3, 0.7, 5),
    dataset_name="manipulated",
    label_dict=obj2int,
    stat_dir=args.stat_dir
)
manip_pred = load_pred_seg_from_json(action_pred_json_file, "manipulated")
save_json(manip_pred, action_pred_json_file.replace("atomic_operation_eval_results.json", "manipulated_eval_results.json"), "manipulated")
mAP, avg_mAP, mRecall = manip_eval.evaluate(manip_pred)
with open(os.path.join(save_dir, "results.txt"), 'a') as f:
    block = '[RESULTS] Action detection results on {:s}.'.format(manip_eval.dataset_name)
    for tiou, tiou_mAP, tiou_mRecall in zip(manip_eval.tiou_thresholds, mAP, mRecall):
        block += '\n|tIoU = {:.2f}: '.format(tiou)
        block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
        for idx, k in enumerate(manip_eval.top_k):
            block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
    block += '\nAverage mAP: {:>4.2f} (%)\n'.format(avg_mAP*100)
    f.write(block)
manip_eval.plot_ap(save_dir)
manip_eval.plot_recall(save_dir)
manip_eval.get_confusion_matrix(save_dir)

affected_eval = ANETdetection(
    val_dataset.data_lists[2],
    val_dataset.split[0],
    tiou_thresholds=np.linspace(0.3, 0.7, 5),
    dataset_name="affected",
    label_dict=obj2int,
    stat_dir=args.stat_dir
)
affected_pred = load_pred_seg_from_json(action_pred_json_file, "affected")
save_json(affected_pred, action_pred_json_file.replace("atomic_operation_eval_results.json", "affected_eval_results.json"), "affected")
mAP, avg_mAP, mRecall = affected_eval.evaluate(affected_pred)
with open(os.path.join(save_dir, "results.txt"), 'a') as f:
    block = '[RESULTS] Action detection results on {:s}.'.format(affected_eval.dataset_name)
    for tiou, tiou_mAP, tiou_mRecall in zip(affected_eval.tiou_thresholds, mAP, mRecall):
        block += '\n|tIoU = {:.2f}: '.format(tiou)
        block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
        for idx, k in enumerate(affected_eval.top_k):
            block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
    block += '\nAverage mAP: {:>4.2f} (%)\n'.format(avg_mAP*100)
    f.write(block)
affected_eval.plot_ap(save_dir)
affected_eval.plot_recall(save_dir)
affected_eval.get_confusion_matrix(save_dir)
