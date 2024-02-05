import torch
from .nms import batched_nms
    

@torch.no_grad()
def inference_single_prediction(
    test_cfg,
    num_class,
    video_list,
    points, fpn_masks, 
    out_cls_logits, out_offsets
):
    # video_list B (list) [dict]
    # points F (list) [T_i, 4]
    # fpn_masks, out_*: F (List) [B, T_i, C]
    results = []

    # 1: gather video meta information
    vid_idxs = [x['video_id'] for x in video_list]
    vid_fps = [x['fps'] for x in video_list]
    vid_lens = [x['duration'] for x in video_list]
    vid_ft_stride = [x['feat_stride'] for x in video_list]
    vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

    # 2: inference on each single video and gather the results
    # upto this point, all results use timestamps defined on feature grids
    for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
        zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
    ):
        # gather per-video outputs
        cls_logits_per_vid = [x[idx] for x in out_cls_logits]
        offsets_per_vid = [x[idx] for x in out_offsets]
        fpn_masks_per_vid = [x[idx] for x in fpn_masks]
        # inference on a single video (should always be the case)
        results_per_vid, _ = inference_single_video_by_score(
            test_cfg, num_class, points, fpn_masks_per_vid,
            cls_logits_per_vid, offsets_per_vid
        )
        # pass through video meta info
        results_per_vid['video_id'] = vidx
        results_per_vid['fps'] = fps
        results_per_vid['duration'] = vlen
        results_per_vid['feat_stride'] = stride
        results_per_vid['feat_num_frames'] = nframes
        results.append(results_per_vid)

    # step 3: postprocssing
    results, _ = postprocessing_nms(test_cfg, results)

    return results


@torch.no_grad()
def inference_multi_prediction_each(
    test_cfg,
    num_classes,
    video_list,
    points, fpn_masks, 
    out_cls_logits, out_offsets
):
    """
    1) Filtering by scores of each type independently.
    2) NMS by scores of each type independently.
    """
    # points F (list) [T_i, 4]
    # fpn_masks: F (List) [T_i, C]
    # out_*: N (List) [F (List) [B, T_i, C]]
    num_cls_types = len(out_cls_logits)
    results = [[] for _ in range(num_cls_types)]

    # 1: gather video meta information
    vid_idxs = [x['video_id'] for x in video_list]
    vid_fps = [x['fps'] for x in video_list]
    vid_lens = [x['duration'] for x in video_list]
    vid_ft_stride = [x['feat_stride'] for x in video_list]
    vid_ft_nframes = [x['feat_num_frames'] for x in video_list]
    
    # 2: inference on each single video and gather the results
    # upto this point, all results use timestamps defined on feature grids
    for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
        zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
    ):
        # gather per-video outputs
        fpn_masks_per_vid = [x[idx] for x in fpn_masks]

        # for other types, you select and keep the same time-idxs calculated above
        for type_num in range(num_cls_types):
            # gather per-video logits/offsets
            cls_logits_per_vid = [x[idx] for x in out_cls_logits[type_num]]
            offsets_per_vid = [x[idx] for x in out_offsets[type_num]]
            results_per_vid, _ = inference_single_video_by_score(
                test_cfg, num_classes[type_num], points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results[type_num].append(results_per_vid)

    # postprocssing
    # NMS by scores of each type.
    for type_num in range(num_cls_types):
        results[type_num], _ = postprocessing_nms(test_cfg, results[type_num])

    return results


@torch.no_grad()
def inference_multi_prediction_together(
    test_cfg,
    num_classes,
    pivot_cls,
    video_list,
    points, fpn_masks, 
    out_cls_logits, out_offsets
):
    """
    1) Filtering by scores of one specific type > keep the same timesteps for the rest.
    2) NMS by scores of one specific type > keep the same instances for the rest.
    """
    num_cls_types = len(out_cls_logits)
    results = [[] for _ in range(num_cls_types)]

    # 1: gather video meta information
    vid_idxs = [x['video_id'] for x in video_list]
    vid_fps = [x['fps'] for x in video_list]
    vid_lens = [x['duration'] for x in video_list]
    vid_ft_stride = [x['feat_stride'] for x in video_list]
    vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

    # 2: inference on each single video and gather the results
    # upto this point, all results use timestamps defined on feature grids
    for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
        zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
    ):
        # gather per-video outputs
        fpn_masks_per_vid = [x[idx] for x in fpn_masks]

        # gather per-video logits for pivot_cls
        cls_logits_per_vid = [x[idx] for x in out_cls_logits[pivot_cls]]
        offsets_per_vid = [x[idx] for x in out_offsets[pivot_cls]]
        # first pre-nms by score of score_by and get timestep-idxs to keep
        results_per_vid, pt_idxs_all = inference_single_video_by_score(
            test_cfg, num_classes[pivot_cls], points, fpn_masks_per_vid,
            cls_logits_per_vid, offsets_per_vid
        )
        # pass through video meta info
        results_per_vid['video_id'] = vidx
        results_per_vid['fps'] = fps
        results_per_vid['duration'] = vlen
        results_per_vid['feat_stride'] = stride
        results_per_vid['feat_num_frames'] = nframes
        results[pivot_cls].append(results_per_vid)

        # for other types, you select and keep the same time-idxs calculated above
        for type_num in range(num_cls_types):
            if type_num == pivot_cls:
                continue
            # gather per-video logits/offsets
            cls_logits_per_vid = [x[idx] for x in out_cls_logits[type_num]]
            offsets_per_vid = [x[idx] for x in out_offsets[type_num]]
            results_per_vid = inference_single_video_with_pt_idxs(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, pt_idxs_all
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results[type_num].append(results_per_vid)

    # postprocssing
    # First do NMS by scores of score_by
    results[pivot_cls], selected_inds = postprocessing_nms(test_cfg, results[pivot_cls])
    # For other types, keep the same detection idxs calculated above
    for type_num in range(num_cls_types):
        if type_num == pivot_cls:
            continue
        results[type_num] = postprocessing_select(results[type_num], selected_inds)

    return results


@torch.no_grad()
def inference_single_video_by_score(
    test_cfg,
    num_classes,
    points,
    fpn_masks,
    out_cls_logits,
    out_offsets
):
    # points F (list) [T_i, 4]
    # fpn_masks, out_*: F (List) [T_i, C]
    segs_all = []
    scores_all = []
    cls_idxs_all = []
    pt_idxs_all = []

    # loop over fpn levels
    for cls_i, offsets_i, pts_i, mask_i in zip(
            out_cls_logits, out_offsets, points, fpn_masks
        ):
        # normalization for output logits
        pred_prob = (cls_i * mask_i.unsqueeze(-1)).flatten()

        # Apply filtering to make NMS faster following detectron2
        # 1. Keep seg with confidence score > a threshold
        keep_idxs1 = (pred_prob > test_cfg['pre_nms_thresh'])
        pred_prob = pred_prob[keep_idxs1]
        topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

        # 2. Keep top k top scoring boxes only
        num_topk = min(test_cfg['pre_nms_topk'], topk_idxs.size(0))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()

        # fix a warning in pytorch 1.9
        pt_idxs =  torch.div(
            topk_idxs, num_classes, rounding_mode='floor'
        )
        cls_idxs = torch.fmod(topk_idxs, num_classes)

        # 3. gather predicted offsets
        offsets = offsets_i[pt_idxs]
        pts = pts_i[pt_idxs]

        # 4. compute predicted segments (denorm by stride for output offsets)
        seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
        seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
        pred_segs = torch.stack((seg_left, seg_right), -1)

        # 5. Keep seg with duration > a threshold (relative to feature grids)
        seg_areas = seg_right - seg_left
        keep_idxs2 = seg_areas > test_cfg['duration_thresh']

        # *_all : S (filtered # of segments) x 2 / 1
        segs_all.append(pred_segs[keep_idxs2])
        scores_all.append(pred_prob[keep_idxs2])
        cls_idxs_all.append(cls_idxs[keep_idxs2])
        pt_idxs_all.append(pt_idxs[keep_idxs2])

    # cat along the FPN levels (F N_i, C)
    segs_all, scores_all, cls_idxs_all = [
        torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
    ]
    results = {'segments' : segs_all,
                'scores'   : scores_all,
                'labels'   : cls_idxs_all}

    return results, pt_idxs_all


@torch.no_grad()
def postprocessing_nms(test_cfg, results):
    # input : list of dictionary items
    # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
    processed_results = []
    selected_inds = []
    for results_per_vid in results:
        # unpack the meta info
        vidx = results_per_vid['video_id']
        fps = results_per_vid['fps']
        vlen = results_per_vid['duration']
        stride = results_per_vid['feat_stride']
        nframes = results_per_vid['feat_num_frames']
        # 1: unpack the results and move to CPU
        segs = results_per_vid['segments'].detach().cpu()
        scores = results_per_vid['scores'].detach().cpu()
        labels = results_per_vid['labels'].detach().cpu()
        selected_ind = torch.arange(segs.shape[0])
        if test_cfg['nms_method'] != 'none':
            # 2: batched nms (only implemented on CPU)
            pre_segs = segs.clone()
            pre_labels = labels.clone()
            segs, scores, labels = batched_nms(
                segs, scores, labels,
                test_cfg['iou_threshold'],
                test_cfg['min_score'],
                test_cfg['max_seg_num'],
                use_soft_nms = (test_cfg['nms_method'] == 'soft'),
                multiclass = test_cfg['multiclass_nms'],
                sigma = test_cfg['nms_sigma'],
                voting_thresh = test_cfg['voting_thresh']
            )
            diff_mat_seg = torch.sum(pre_segs.unsqueeze(1) - segs, dim=-1)  # (#pre_segs, #segs)
            diff_mat_label = pre_labels.unsqueeze(1) - labels  # (#pre_labels, #labels)
            diff_mat = diff_mat_seg + diff_mat_label
            # inds of matched pair of pre_segs and segs 
            eq_inds = torch.nonzero(diff_mat == 0)  # (#segs, 2)
            selected_ind = eq_inds[eq_inds[:, 1].sort()[1]][:, 0]  # (#segs, 1)
        selected_inds.append(selected_ind)
        # 3: convert from feature grids to seconds
        if segs.shape[0] > 0:
            segs = (segs * stride + 0.5 * nframes) / fps
            # truncate all boundaries within [0, duration]
            segs[segs<=0.0] *= 0.0
            segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
        
        # 4: repack the results
        processed_results.append(
            {'video_id' : vidx,
                'segments' : segs,
                'scores'   : scores,
                'labels'   : labels}
        )

    return processed_results, selected_inds


@torch.no_grad()
def inference_single_video_with_pt_idxs(
    points,
    fpn_masks,
    out_cls_logits,
    out_offsets,
    pt_idxs_all
):
    """
    Reduce detections by keeping only ones on given point indexes. 
    """
    # points F (list) [T_i, 4]
    # fpn_masks, out_*: F (List) [T_i, C]
    segs_all = []
    scores_all = []
    cls_idxs_all = []

    # loop over fpn levels
    for cls_i, offsets_i, pts_i, mask_i, pt_idxs in zip(
            out_cls_logits, out_offsets, points, fpn_masks, pt_idxs_all
        ):
        # gather predicted offsets
        offsets = offsets_i[pt_idxs]
        pts = pts_i[pt_idxs]

        # compute predicted segments (denorm by stride for output offsets)
        seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
        seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
        pred_segs = torch.stack((seg_left, seg_right), -1)

        # mask out the padding
        pred_prob_all = (cls_i * mask_i.unsqueeze(-1))
        
        # # TODO if the same points are included more than once, class end up being same here
        # pred_prob, cls_idxs = torch.max(pred_prob_all[pt_idxs], dim=-1)
        
        # if the same points are included more than once, put different class in the descending order of the score.
        # sort classificaton vector on each time point
        sorted_pred_prob_all, sorted_cls_idxs = torch.sort(pred_prob_all, dim=1, descending=True)
        # sort extracted points (e.g., [2, 1, 0, 1, ...] -> [0, 1, 1, 2, ...])
        pt_idxs_sort, tmp_indices = torch.sort(pt_idxs, stable=True)
        # count duplicated points (e.g., [0: 1, 1: 2, 2: 1, ...])
        _, cnts = torch.unique(pt_idxs_sort, sorted=True, return_counts=True)  
        # number each dupliate (e.g., [0, 0, 1, 0, ...]) 
        h_indices = torch.concat([torch.linspace(0, min(cnt, cls_i.shape[1] - 1), cnt).to(torch.long) for cnt in cnts]).to(tmp_indices.device)
        # put the order back to original (e.g., [0, 0, 0, 1, ...])
        h_indices = h_indices[torch.sort(tmp_indices)[1]]   
        # extract probability (if the same point appear multiple times, extract the next max prob.)
        pred_prob = sorted_pred_prob_all[pt_idxs, h_indices]
        # extract corresponding class index
        cls_idxs = sorted_cls_idxs[pt_idxs, h_indices]

        # *_all : N (filtered # of segments) x 2 / 1
        segs_all.append(pred_segs)
        scores_all.append(pred_prob)
        cls_idxs_all.append(cls_idxs)

    # cat along the FPN levels (F N_i, C)
    segs_all, scores_all, cls_idxs_all = [
        torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
    ]
    results = {'segments' : segs_all,
                'scores'   : scores_all,
                'labels'   : cls_idxs_all}

    return results
    
    
@torch.no_grad()
def postprocessing_select(results, selected_inds):
    # (1) push to CPU; (2) selection; (3) convert to actual time stamps
    processed_results = []
    for i, results_per_vid in enumerate(results):
        # unpack the meta info
        vidx = results_per_vid['video_id']
        fps = results_per_vid['fps']
        vlen = results_per_vid['duration']
        stride = results_per_vid['feat_stride']
        nframes = results_per_vid['feat_num_frames']
        # unpack the results and move to CPU and select segments
        segs = results_per_vid['segments'].detach().cpu()[selected_inds[i]]
        scores = results_per_vid['scores'].detach().cpu()[selected_inds[i]]
        labels = results_per_vid['labels'].detach().cpu()[selected_inds[i]]
        # convert from feature grids to seconds
        if segs.shape[0] > 0:
            segs = (segs * stride + 0.5 * nframes) / fps
            # truncate all boundaries within [0, duration]
            segs[segs<=0.0] *= 0.0
            segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
        
        # 4: repack the results
        processed_results.append(
            {'video_id' : vidx,
                'segments' : segs,
                'scores'   : scores,
                'labels'   : labels}
        )

    return processed_results