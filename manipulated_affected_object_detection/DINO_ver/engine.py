# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import numpy as np

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.finebio_eval import HandObjEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            
        # for n, p in model.named_parameters():
        #     if p.requires_grad and p.grad is not None:
        #         print("%s: %.4f" % (n, p.grad.norm(2)))

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
        
    # coco evaluator
    # coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    coco_evaluator = None
    
    # hand object interaction evalator
    eval_constraints = ['handstate', 'manipulated_bbox', 'manipulated_bbox&class']
    if args.predict_affectedobj:
        eval_constraints += ['manipulated_bbox&class_affected_bbox', 'manipulated_bbox&class_affected_bbox&class']
    handobj_evaluator = HandObjEvaluator(base_ds, eval_constraints=eval_constraints)

    _cnt = 0
    output_state_dict = {} # for debug only
    num_images = len(data_loader)
    num_classes = model.num_classes
    hand_ids = [data_loader.dataset.lhand_id, data_loader.dataset.rhand_id]
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        assert len(targets) == 1, "batch size should be 1 in inference."
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
        # update handobj results
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
            
        for j in range(1, num_classes):
            inds = torch.nonzero(labels == j).view(-1)
                
            if inds.numel() > 0:
                thres_score = args.thres_obj
                if j in hand_ids:
                    thres_score = args.thres_hand
                inds = inds[scores[inds] > thres_score]
                cls_scores = scores[inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = boxes[inds]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), hand_state_indices[inds], hand_vector[inds], manipulated_state_indices[inds], affecting_state_indices[inds], affected_state_indices[inds]), 1)
                if j not in hand_ids:
                    cls_dets[:, 6:9] = manip_vector[inds]
                cls_dets = cls_dets[order].cpu().numpy()
            else:
                cls_dets = np.transpose(np.array([[],[],[],[],[],[],[],[],[],[],[]]), (1,0))
            handobj_evaluator.update_result_dict(targets[0]['image_id'].cpu().numpy()[0], j, cls_dets)
        handobj_evaluator.update_gt_dict({k: v.cpu().numpy() for k, v in targets[0].items()})
        
        if args.save_results:
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        catIds = coco_evaluator.coco_eval[coco_evaluator.iou_types[0]].params.catIds
        catNames = [coco_evaluator.coco_gt.cats[catId]['name'] for catId in catIds]
        for catId, catName in zip(catIds, catNames):
            print(catName)
            coco_evaluator.summarize_one_cls(catId)
    
    # evaluation for hand object detection 
    # mean (detection aps + hand-obj interaction aps)
    all_avg_ap = handobj_evaluator.evaluate_detections()
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats["handobj_mean_ap"] = all_avg_ap
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()


    return stats, None

