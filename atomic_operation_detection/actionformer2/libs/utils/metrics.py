# Modified from official EPIC-Kitchens action detection evaluation code
# see https://github.com/epic-kitchens/C2-Action-Detection/blob/master/EvaluationCode/evaluate_detection_json_ek100.py
import os
import copy
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def remove_duplicate_annotations(ants):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        valid = True
        for p_event in valid_events:
            if ((s == p_event['segment'][0]) and 
                (e == p_event['segment'][1]) and
                (l == p_event['label_id'])):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        # filter based on split
        if (split is not None) and v['subset'].lower() != split:
            continue
        # remove duplicated instances
        ants = remove_duplicate_annotations(v['annotations'])
        # ants = v["annotations"]
        # video id
        vids += [k] * len(ants)
        # for each event, grab the start/end time and label
        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]

    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        # video id
        vids += [k] * len(v)
        # for each event
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    # move to pd dataframe
    pred_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base

def load_gt_seg_from_datalist(data_list, label='label_id', label_offset=0):
    vids, starts, stops, labels = [], [], [], []
    for data in data_list:
        ants = []
        for i in range(len(data["segments"])):
            ants.append({
                "segment": [data["segments"][i][0], data["segments"][i][1]],
                "label_id": data["labels"][i]
            })
        ants = remove_duplicate_annotations(ants)
        # video id
        vids += [data["id"]] * len(ants)
        # for each event, grab the start/end time and label
        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                # offset the labels by label_offset
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                # load label_id directly
                label_id = int(event[label])
            labels += [label_id]

    # move to pd dataframe
    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


class ANETdetection(object):
    """Adapted from https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py"""

    def __init__(
        self,
        ant,
        split=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        top_k=(1, 5),
        label='label_id',
        label_offset=0,
        num_workers=8,
        dataset_name=None,
        label_dict=None,
        stat_dir=None
    ):

        self.tiou_thresholds = tiou_thresholds
        self.top_k = top_k
        self.ap = None
        # matched ground truth id for each ground truth
        self.matched_gt_id = None
        self.prediction = None
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant).replace('.json', '')

        # Import ground truth and predictions
        self.split = split
        if type(ant) == str:
            self.ground_truth = load_gt_seg_from_json(
                ant, split=self.split, label=label, label_offset=label_offset)
        elif type(ant) == tuple:
            self.ground_truth = load_gt_seg_from_datalist(ant, label=label, label_offset=label_offset)

        # remove labels that does not exists in gt
        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.ground_truth['label'] = self.ground_truth['label'].replace(self.activity_index)
        self.ground_truth['gt-id'] = range(len(self.ground_truth))

        # reversed dict of self.activity_index (index > original label id)
        self.activity_index_reverse = dict(zip(self.activity_index.values(), self.activity_index.keys()))

        # dict mapping original label index(=keys of self.activity_index) to label name
        self.int2label = None
        if label_dict:
            self.int2label = dict(zip(label_dict.values(), label_dict.keys()))
            
        self.stat_dir = stat_dir

    def _get_label2cnt(self, subset="all"):
        """Get frequency for each label.
        """
        if self.stat_dir and os.path.exists(self.stat_dir):
            type_name = self.dataset_name
            path = os.path.join(self.stat_dir, f"label2cnt_{type_name.replace('left_', '').replace('right_', '')}.csv")
            data = pd.read_csv(path)
            data = data[(data["subset"] == subset) & (data["type_name"] == type_name)]
            data = data.drop(["subset", "type_name"], axis=1).reset_index(drop=True)
            label2cnt = data.loc[0].to_dict()
            return label2cnt
        else:
            return None
    
    def _get_label2duration(self, subset='all'):
        """Get mean duration for each label in ground truth
        """
        if self.stat_dir and os.path.exists(self.stat_dir):
            type_name = self.dataset_name
            path = os.path.join(self.stat_dir, f"label2duration_{type_name.replace('left_', '').replace('right_', '')}.csv")
            data = pd.read_csv(path)
            data = data[(data["subset"] == subset) & (data["type_name"] == type_name)]
            data = data.drop(["subset", "type_name"], axis=1).reset_index(drop=True)
            label2duration = data.loc[0].to_dict()
            return label2duration
        else:
            return None

    # def _get_grondtruth_with_label(self, groundtruth_by_label, label_name, cidx):
    #     """Get all predicitons of the given label. Return empty DataFrame if there
    #     is no predcitions with the given label.
    #     """
    #     try:
    #         res = groundtruth_by_label.get_group(cidx).reset_index(drop=True)
    #         return res
    #     except:
    #         print('Warning: No groundtruths of label \'%s\' were provdied.' % label_name)
    #         return pd.DataFrame()
    
    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        """Computes average precision for each class in the subset.
        """
        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')
        
        ap = np.zeros((len(self.tiou_thresholds), len(ground_truth_by_label.groups)))
        matched_gt_id = -np.ones((len(self.tiou_thresholds), len(preds)))

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, self.activity_index_reverse[cidx], cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for cidx in ground_truth_by_label.groups)

        for i, cidx in enumerate(ground_truth_by_label.groups):
            ap[:, cidx], matched_this_cls_gt_id, this_cls_prediction_ids = results[i]
            matched_gt_id[:, this_cls_prediction_ids] = matched_this_cls_gt_id

        return ap, matched_gt_id

    def wrapper_compute_topkx_recall(self, preds):
        """Computes Top-kx recall for each class in the subset.
        """
        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        recall = np.zeros((len(self.tiou_thresholds), len(self.top_k), len(ground_truth_by_label.groups)))

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_topkx_recall_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, self.activity_index_reverse[cidx], cidx),
                tiou_thresholds=self.tiou_thresholds,
                top_k=self.top_k,
            ) for cidx in ground_truth_by_label.groups)

        for i, _ in enumerate(ground_truth_by_label.groups):
            recall[..., i] = results[i]

        return recall
    
    def add_fp_match(self, preds):
        """Add the matched gt-id for fp to self.matched_gt_id and return the new matrix. 
        """
        matched_gt_id = copy.deepcopy(self.matched_gt_id)
        ground_truth_gbvn = self.ground_truth.groupby('video-id')
        for tidx, tiou in enumerate(self.tiou_thresholds):
            fp_prediction = preds[matched_gt_id[tidx] == -1]
            fp_prediction = fp_prediction.sort_values(by='video-id').reset_index(drop=True)
            
            current_video_id = None
            
            for idx, this_pred in fp_prediction.iterrows():
                if this_pred["video-id"] != current_video_id:
                    try:
                        this_gt = ground_truth_gbvn.get_group(this_pred['video-id']).reset_index()
                    except:
                        continue
                    current_video_id = this_pred["video-id"]
                
                tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
                # We would like to retrieve the predictions with highest tiou score.
                gt_with_max_tiou_label = this_gt.loc[tiou_arr.argmax()]['label']
                top_tiou = tiou_arr.max()
                this_pred_label = this_pred['label']
                
                # if location is ok but label is different
                if top_tiou >= tiou and gt_with_max_tiou_label != this_pred_label:
                    matched_gt_id[tidx, this_pred["pred-id"]] = this_gt.loc[tiou_arr.argmax()]["gt-id"]
        return matched_gt_id


    def evaluate(self, preds, verbose=True):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        preds can be (1) a pd.DataFrame; or (2) a json file where the data will be loaded;
        or (3) a python dict item with numpy arrays as the values
        """

        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            # move to pd dataframe
            # did not check dtype here, can accept both numpy / pytorch tensors
            preds = pd.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
        # always reset ap and matched_gt_id
        self.ap = None
        self.matched_gt_id = None

        # make the label ids consistent
        # add novel labels which don't appear in gt but appear in prediction
        pred_unique_labels = set(preds['label'].unique())
        for label in (pred_unique_labels - set(self.activity_index.keys())):
            new_idx = len(self.activity_index)
            self.activity_index[label] = new_idx
            self.activity_index_reverse[new_idx] = label
        preds['label'] = preds['label'].replace(self.activity_index)
        preds['pred-id'] = range(len(preds))
        self.prediction = preds

        # compute mAP
        self.ap, self.matched_gt_id = self.wrapper_compute_average_precision(preds)
        self.recall = self.wrapper_compute_topkx_recall(preds)
        mAP = self.ap.mean(axis=1)
        mRecall = self.recall.mean(axis=2)
        average_mAP = mAP.mean()
        
        if verbose:
            # print the results
            print('[RESULTS] Action detection results on {:s}.'.format(
                self.dataset_name)
            )
            block = ''
            for tiou, tiou_mAP, tiou_mRecall in zip(self.tiou_thresholds, mAP, mRecall):
                block += '\n|tIoU = {:.2f}: '.format(tiou)
                block += 'mAP = {:>4.2f} (%) '.format(tiou_mAP*100)
                for idx, k in enumerate(self.top_k):
                    block += 'Recall@{:d}x = {:>4.2f} (%) '.format(k, tiou_mRecall[idx]*100)
            print(block)
            print('Average mAP: {:>4.2f} (%)'.format(average_mAP*100))

        # return the results
        return mAP, average_mAP, mRecall
    
    def plot_ap(self, save_dir, tiou_thresholds=[0.3, 0.5, 0.7]):
        """plot AP for each label
        Args:
            save_dir(str): path to save plot images
            tiou_thresholds(list): tIoU for AP
        """
        if self.ap is None:
            print("APs are not calculated yet. Execute evaluate() first.")
            return
        
        if self.int2label is None:
            print('Label name is not specified.')
            return
        
        ground_truth_by_label = self.ground_truth.groupby('label')
        
        for tidx, tiou in enumerate(self.tiou_thresholds):
            if tiou not in tiou_thresholds:
                continue
            aps = self.ap[tidx]
            left = np.arange(1, len(aps) + 1)
            heights = aps
            labels = [self.activity_index_reverse[cidx] for cidx in ground_truth_by_label.groups]
            # sort by AP
            sort_idxs = np.argsort(heights)[::-1]
            # calculate frequency
            label2cnt = self._get_label2cnt()  # label frequency in all data.
            # additional info (duration)
            label2info = self._get_label2duration() # label average duration in all data
            info_name = "Avg duration"
            # idx > name
            labels = list(map(lambda idx: self.int2label[int(idx)], labels))
            # save text
            save_txt_file = os.path.join(save_dir, f"{self.dataset_name}_tIoU{tiou}_ap.txt")
            with open(save_txt_file, 'w') as f:
                for idx in sort_idxs:
                    f.write(f"{labels[idx]} {heights[idx]}\n")
            # visualization
            if len(labels) < 100:
                fig, ax1 = plt.subplots(figsize=(10, 8))
            else:
                fig, ax1 = plt.subplots(figsize=(60 * (len(labels) // 100), 8))
            ax1.bar(left, [heights[idx] for idx in sort_idxs], tick_label=[labels[idx] for idx in sort_idxs], align="center", label="AP", color="orange")
            # legend setting
            handler, label = ax1.get_legend_handles_labels()
            # adjust range
            ax1.set_ylim([0, 0.1 + max(heights)])
            # label setting
            ax1.set_xlabel("Label", fontsize=20)
            ax1.tick_params(axis="x", labelrotation=90)
            ax1.set_ylabel("AP", fontsize=20)
            ax1.tick_params(labelsize=20)
            if label2cnt:
                ax2 = ax1.twinx()
                ax2.plot(left, [label2cnt[labels[idx]] for idx in sort_idxs], label="Frequency", color="green")
                handler2, label2 = ax2.get_legend_handles_labels()
                handler += handler2
                label += label2
                ax2.set_ylim([0, 100 + max(label2cnt.values())])
                ax2.set_ylabel("Frequency", fontsize=20, color="green")
                ax2.tick_params(labelsize=20, colors="green")
                ax2.spines["right"].set_color('green')
            if label2info:
                ax3 = ax1.twinx()
                ax3.plot(left, [label2info[labels[idx]] for idx in sort_idxs], label=info_name, color="blue")
                handler3, label3 = ax3.get_legend_handles_labels()
                handler += handler3 
                label += label3
                ax3.set_ylim([0, 1.1 * max(label2info.values())])
                ax3.set_ylabel(info_name, fontsize=20, color="blue")
                ax3.tick_params(labelsize=20, colors="blue")
                ax3.spines["right"].set_color('blue')
                ax3.spines["right"].set_position(("outward", 140))
            ax1.legend(handler, label, fontsize=16, loc="lower center", bbox_to_anchor=(.5, 1.), ncol=3)
            plt.title(f"{self.dataset_name} AP", fontsize=20, x=0.5, y=1.1)
            save_img_file = os.path.join(save_dir, f"{self.dataset_name}_tIoU{tiou}_ap.png")
            fig.savefig(save_img_file, bbox_inches="tight")
        plt.clf()
        plt.close()
    
    def plot_recall(self, save_dir, tiou_thresholds=[0.3, 0.5, 0.7], top_k=[1]):
        """plot recall for each label
        Args:
            save_dir(str): path to save plot images
            tiou_thresholds(list): tIoU for Recall
            top_k(list): what times the number of prediction is as big as the number of gt. 
        """
        if self.recall is None:
            print("APs are not calculated yet. Execute evaluate() first.")
            return
        
        if self.int2label is None:
            print('Label name is not specified.')
            return
            
        ground_truth_by_label = self.ground_truth.groupby('label')
        
        for tidx, tiou in enumerate(self.tiou_thresholds):
            if tiou not in tiou_thresholds:
                    continue
            for kidx, k in enumerate(self.top_k):
                if k not in top_k:
                    continue
                recalls = self.recall[tidx][kidx]
                left = np.arange(1, len(recalls) + 1)
                heights = recalls
                labels = [self.activity_index_reverse[cidx] for cidx in ground_truth_by_label.groups]
                # sort by AP
                sort_idxs = np.argsort(heights)[::-1]
                # calculate frequency
                label2cnt = self._get_label2cnt()
                # additional info (duration)
                label2info = self._get_label2duration()
                info_name = "Avg duration"
                # idx > name
                labels = list(map(lambda idx: self.int2label[int(idx)], labels))
                # save text
                save_txt_file = os.path.join(save_dir, f"{self.dataset_name}_tIoU{tiou}_top{k}-recall.txt")
                with open(save_txt_file, 'w') as f:
                    for idx in sort_idxs:
                        f.write(f"{labels[idx]} {heights[idx]}\n")
                # visualization
                if len(labels) < 100:
                    fig, ax1 = plt.subplots(figsize=(10, 8))
                else:
                    fig, ax1 = plt.subplots(figsize=(60 * (len(labels) // 100), 8))
                ax1.bar(left, [heights[idx] for idx in sort_idxs], tick_label=[labels[idx] for idx in sort_idxs], align="center", label="AP", color="orange")
                # legend setting
                handler, label = ax1.get_legend_handles_labels()
                # adjust range
                ax1.set_ylim([0, 0.1 + max(heights)])
                # label setting
                ax1.set_xlabel("Label", fontsize=20)
                ax1.tick_params(axis="x", labelrotation=90)
                ax1.set_ylabel("AP", fontsize=20)
                ax1.tick_params(labelsize=20)
                if label2cnt:
                    ax2 = ax1.twinx()
                    ax2.plot(left, [label2cnt[labels[idx]] for idx in sort_idxs], label="Frequency", color="green")
                    handler2, label2 = ax2.get_legend_handles_labels()
                    handler += handler2
                    label += label2
                    ax2.set_ylim([0, 100 + max(label2cnt.values())])
                    ax2.set_ylabel("Frequency", fontsize=20, color="green")
                    ax2.tick_params(labelsize=20, colors="green")
                    ax2.spines["right"].set_color('green')
                if label2info:
                    ax3 = ax1.twinx()
                    ax3.plot(left, [label2info[labels[idx]] for idx in sort_idxs], label=info_name, color="blue")
                    handler3, label3 = ax3.get_legend_handles_labels()
                    handler += handler3
                    label += label3
                    ax3.set_ylim([0, 1.1 * max(label2info.values())])
                    ax3.set_ylabel(info_name, fontsize=20, color="blue")
                    ax3.tick_params(labelsize=20, colors="blue")
                    ax3.spines["right"].set_color('blue')
                    ax3.spines["right"].set_position(("outward", 140))
                ax1.legend(handler, label, fontsize=16, loc="lower center", bbox_to_anchor=(.5, 1.), ncol=3)
                plt.title(f"{self.dataset_name} Recall", fontsize=20, x=0.5, y=1.1)
                save_file = os.path.join(save_dir, f"{self.dataset_name}_tIoU{tiou}_top{k}-recall.png")
                fig.savefig(save_file, bbox_inches="tight")
        plt.clf()
        plt.close()

    def get_confusion_matrix(self, save_dir, tiou_thresholds=[0.3, 0.5, 0.7]):
        """Get confusion matrix.
        One or zero prediction for one gt.
        Args:
            save_dir(str): path to save confusion matrix images
            tiou_thresholds(list): tIoU threshold to judge match
        """
        if self.matched_gt_id is None:
            print("Prediction and ground truth are not matched yet. Execute evaluate() first.")
            return
        # Besides tp matches, find matched gt id for fp prediction and update matched_gt_id.
        matched_gt_id = self.add_fp_match(self.prediction)
        
        for tidx, tiou in enumerate(self.tiou_thresholds):
            if tiou not in tiou_thresholds:
                continue
            # extract matched pair
            matched_mask = (matched_gt_id[tidx] != -1)
            gt_ids = matched_gt_id[tidx][matched_mask]
            gt_labels = self.ground_truth.loc[gt_ids, "label"].values
            pred_labels = self.prediction["label"][matched_mask].values
            column_labels = np.concatenate(
                [sorted(np.unique(gt_labels)), 
                 sorted(np.setdiff1d(pred_labels, gt_labels))])
            row_labels = sorted(np.unique(gt_labels))
            # calculate confusion matrix
            cfmat = confusion_matrix(gt_labels, pred_labels, labels=column_labels)
            cfmat = cfmat[:len(row_labels), :]
            sum_ = np.clip(np.sum(cfmat, axis=1, keepdims=True), 1, None) 
            cfmat = cfmat / sum_  # normalization
            # create dataframe to visualize
            index = [self.activity_index_reverse[i] for i in row_labels]
            columns = [self.activity_index_reverse[i] for i in column_labels]
            data = pd.DataFrame(data=cfmat,
                                index=index,
                                columns=columns)
            if self.int2label:
                data.index = list(map(lambda x: self.int2label[x], index))
                data.columns = list(map(lambda x: self.int2label[x], columns))
            # visualization
            if len(self.activity_index) < 100:
                fig = plt.figure(figsize=(14, 12))
            else:
                # if confusion matrix is so large, save dataframe instead of image
                save_file = os.path.join(save_dir, f"{self.dataset_name}_tIoU{tiou}_cfmat.csv")
                data.to_csv(save_file)
                continue
            plt.ylim(0, data.shape[0])
            sns.heatmap(data, cmap="Blues", annot=True, fmt=".2f")
            plt.xlabel("Prediction", fontsize=20)
            plt.ylabel("Ground Truth", fontsize=20)
            plt.tick_params(labelsize=20)
            save_file = os.path.join(save_dir, f"{self.dataset_name}_tIoU{tiou}_cfmat.pdf")
            fig.savefig(save_file, bbox_inches="tight")
        plt.clf()
        plt.close()


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    matched_gt_id : np.array [#tiou_thresholds, #prediction]
        Matched gt id for each prediction.
    this_cls_prediction_ids: np.array [#prediction]
        List of prediction ids for the class.
    """
    ap = np.zeros(len(tiou_thresholds))
    matched_gt_id = -np.ones((len(tiou_thresholds), len(prediction)))
    if prediction.empty:
        return ap, matched_gt_id, []

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly ground truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                matched_gt_id[tidx, idx] = this_gt.loc[jdx]['gt-id']
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap, matched_gt_id, prediction['pred-id'].values


def compute_topkx_recall_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5),
    top_k=(1, 5),
):
    """Compute recall (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    top_k: tuple, optional
        Top-kx results of a action category where x stands for the number of 
        instances for the action category in the video.
    Outputs
    -------
    recall : float
        Recall score.
    """
    if prediction.empty:
        return np.zeros((len(tiou_thresholds), len(top_k)))

    # Initialize true positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(top_k)))
    n_gts = 0

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    prediction_gbvn = prediction.groupby('video-id')

    for videoid, _ in ground_truth_gbvn.groups.items():
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        n_gts += len(ground_truth_videoid)
        try:
            prediction_videoid = prediction_gbvn.get_group(videoid)
        except Exception as e:
            continue

        this_gt = ground_truth_videoid.reset_index()
        this_pred = prediction_videoid.reset_index()

        # Sort predictions by decreasing score order.
        score_sort_idx = this_pred['score'].values.argsort()[::-1]
        top_kx_idx = score_sort_idx[:max(top_k) * len(this_gt)]
        tiou_arr = k_segment_iou(this_pred[['t-start', 't-end']].values[top_kx_idx],
                                 this_gt[['t-start', 't-end']].values)
            
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for kidx, k in enumerate(top_k):
                tiou = tiou_arr[:k * len(this_gt)]
                tp[tidx, kidx] += ((tiou >= tiou_thr).sum(axis=0) > 0).sum()

    recall = tp / n_gts

    return recall


def find_match_pred(ground_truth, prediction, tiou_thresholds):
    """Find matched prediction for each ground truth (if any).
    "match" is defined as groundtruth and prediction temporally overlapping at a higher rate than threshold.
    Parameters.
    One or zero prediction for one ground truth.
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_threshold : float, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    matched_pred_label : np.array (#thres, #ground_truth)
        Matched pred label for each ground truth (if it doesn't exist, -1) at each tIoU
    """
    # matched prediction label for each ground truth
    matched_pred_label = -np.ones((len(tiou_thresholds), len(ground_truth)))
    # sort by prediction score
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    for idx, this_pred in prediction.iterrows():
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               ground_truth[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    break
                # when matched prediction had already been found for the gt.
                if matched_pred_label[tidx, jdx] >= 0:
                    continue
                # Assign matched prediction for the gt.
                matched_pred_label[tidx, jdx] = this_pred['label']
                break
    
    return matched_pred_label

def k_segment_iou(target_segments, candidate_segments):
    return np.stack(
        [segment_iou(target_segment, candidate_segments) \
            for target_segment in target_segments]
    )


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
