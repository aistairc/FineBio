import math
import os
import copy
import json
import numpy as np
from glob import glob

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from .grpe import GRPENetwork

from .prediction_fuser import PredictionFuser


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,               # other cfg for testing
        apply_graph_modeling=False  # whether apply operation graph modeling
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )
        
        # whether apply grpe
        self.apply_graph_modeling=apply_graph_modeling
        if self.apply_graph_modeling:
            self.op_grpe_adj_fr_num = 10
            # GRPE architecture config
            self.op_grpe_input_dim = self.num_classes
            self.op_grpe_hidden_dim = 200
            self.op_grpe_num_layers = 2
            self.op_grpe_num_heads = 4
            self.op_grpe_dropout = 0.
            self.op_grpe_att_dropout = 0.
            self.op_grpe_num_edge_type = 1  # not necessary
            self.op_grpe_max_hop = self.op_grpe_adj_fr_num * 2
            self.op_grpe_use_independent_token = False
            self.op_grpe_perturb_noise = 0.
            self.op_grpe_num_last_mlp = 0
            
            self.op_grpe_net = GRPENetwork(self.op_grpe_input_dim, self.op_grpe_hidden_dim, self.op_grpe_input_dim,
                                        num_layer=self.op_grpe_num_layers, nhead=self.op_grpe_num_heads,
                                        dropout=self.op_grpe_dropout, attention_dropout=self.op_grpe_att_dropout,
                                        max_hop=self.op_grpe_max_hop, num_edge_type=self.op_grpe_num_edge_type, 
                                        use_independent_token=self.op_grpe_use_independent_token,
                                        perturb_noise=self.op_grpe_perturb_noise, num_last_mlp=self.op_grpe_num_last_mlp,
                                        )
            self.refine_op_cls_head = PtTransformerClsHead(
                                self.op_grpe_input_dim, 
                                head_dim, self.num_classes,
                                kernel_size=head_kernel_size,
                                prior_prob=self.train_cls_prior_prob,
                                with_ln=head_with_ln,
                                num_layers=head_num_layers
                            )
            self.graph_loss_weight = 0.3
            
        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini//-batch)
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)
        
        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        
        if self.apply_graph_modeling:
            refined_cls_logits = self.post_op_grpe(
                video_list, points, out_offsets, out_cls_logits, fpn_feats, fpn_masks
            )
            refined_cls_logits = [x.permute(0, 2, 1) for x in refined_cls_logits]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            
            if self.apply_graph_modeling:
                refined_losses = self.losses(
                    fpn_masks, 
                    refined_cls_logits, out_offsets,
                    gt_cls_labels, gt_offsets
                )
                losses['refined_cls_loss'] = refined_losses['cls_loss']
                # losses['refined_reg_loss'] = refined_losses['reg_loss']
                losses['final_loss'] = losses['final_loss'] + refined_losses['cls_loss']
            
            return losses

        else:
            # !!!!return logits after sigmoid!!!!
            if self.apply_graph_modeling:
                refined_cls_logits = [x.sigmoid() for x in refined_cls_logits]
                return points, fpn_masks, refined_cls_logits, out_offsets
            else:
                out_cls_logits = [x.sigmoid() for x in out_cls_logits]
                return points, fpn_masks, out_cls_logits, out_offsets

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}
    
    def post_op_grpe(self, video_list, points, init_offsets, init_cls_logits, fpn_feats, fpn_masks):   
        level_splits = [x.shape[2] for x in fpn_feats]   
        # F List[B, T_i, C] -> [B, FT, C]
        fpn_feats = [x.permute(0, 2, 1) for x in fpn_feats]
        fpn_feats = torch.cat(fpn_feats, dim=1)
        valid_masks = torch.cat(fpn_masks, dim=1)
        init_offsets = torch.cat(init_offsets, dim=1).detach().clone()
        # calculate segment
        num_vid = init_offsets.shape[0]
        num_pts = init_offsets.shape[1]
        points_cat = torch.cat(points).detach().unsqueeze(0).repeat(num_vid, 1, 1)
        init_seg_left = points_cat[:, :, 0] - init_offsets[:, :, 0] * points_cat[:, :, 3]
        init_seg_right = points_cat[:, :, 0] + init_offsets[:, :, 1] * points_cat[:, :, 3]
        # apply graph processing per video
        init_op_logits = torch.cat(init_cls_logits, dim=1)
        refined_feats = []
        for i in range(num_vid):
            # sort by time
            temp_sort_idxs = torch.from_numpy(np.lexsort((init_seg_left[i].cpu().numpy(), init_seg_right[i].cpu().numpy()))).to(points_cat.device)
            valid_mask_per_vid = valid_masks[i][temp_sort_idxs]
            # extract only valid operation segments and sampling by score
            valid_idxs = temp_sort_idxs[valid_mask_per_vid]
            
            # split operation into entities 
            op_logits = init_op_logits[i, valid_idxs]
            num_nodes = len(op_logits)
            
            # create distance matrix
            distance_mat = torch.zeros((1, num_nodes, num_nodes), device=self.device, dtype=torch.int64)
            seg_dist_mat = (torch.arange(num_nodes).unsqueeze(0).repeat(num_nodes, 1) - torch.arange(num_nodes).unsqueeze(-1).repeat(1, num_nodes)).to(self.device)
            intra_seg_mask = (seg_dist_mat == 0)
            inter_seg_mask = ~intra_seg_mask
            distance_mat[0] += torch.tril(seg_dist_mat + self.op_grpe_adj_fr_num + 1, diagonal=-1) * inter_seg_mask  # pre-segment
            distance_mat[0] += torch.triu(seg_dist_mat + self.op_grpe_adj_fr_num, diagonal=1) * inter_seg_mask  # post-segment
            distance_mat[(distance_mat > self.op_grpe_max_hop) | (distance_mat < 0)] = -1
            # for e in range(6):
            #     print(','.join(map(str, distance_mat[0][e].tolist()[:50])))
            
            # create edge attr matrix
            # 0: self, 1: connected
            edge_attr_mat = -torch.ones((1, num_nodes, num_nodes), device=self.device, dtype=torch.int64)
            edge_attr_mat[distance_mat != -1] = 1
            edge_attr_mat[0, torch.arange(num_nodes), torch.arange(num_nodes)] = 0
            # for e in range(6):
            #     print(','.join(map(str, edge_attr_mat[0][e].tolist()[:50])))

            # GRPE
            op_feats = self.op_grpe_net(
                    op_logits.unsqueeze(0),
                    mask=torch.zeros((1, num_nodes), device=op_logits.device, dtype=torch.bool),
                    distance_mat=distance_mat,
                    edge_attr_mat=edge_attr_mat)[0]
            pad_cls_feats = init_op_logits[i].detach().clone()
            pad_cls_feats[valid_idxs] = op_feats
            refined_feats.append(torch.split(pad_cls_feats, level_splits))
        # classification / regression as refinement
        # convert dim order (B List[F T_i C] -> F List [B C T_i]) to fit in cls_head / reg_head
        convert_refined_feats = []
        for i in range(len(level_splits)):
            refined_feats_i = [x[i].T for x in refined_feats]
            refined_feats_i = torch.stack(refined_feats_i)
            convert_refined_feats.append(refined_feats_i)
        # pass converted feats to cls_head / reg_head
        refined_cls_logits = self.refine_op_cls_head(convert_refined_feats, [x.unsqueeze(1) for x in fpn_masks])
        return refined_cls_logits  # F List[B, C, T_i]


@register_meta_arch("MultiPredictionLocPointTransformer")
class MultiPredictionPtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
        with_op_pred=False,    # whether predict operations by three entity predictions
        op_pred_method="fuse", # how to fuse entity predictions into operation prediction
        apply_op_graph_modeling=False  # whether apply operation graph modeling
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # number of types to describe action (e.g., verb, noun etc)
        self.num_cls_types = len(self.num_classes)

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # classfication and regerssion heads
        self.cls_head = nn.ModuleList([
            PtTransformerClsHead(
                fpn_dim, head_dim, self.num_classes[i],
                kernel_size=head_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=train_cfg['head_empty_cls']
            ) for i in range(self.num_cls_types)
        ])
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )
        
        # whether three heads (verb/manipulated/affected) are fused into atomic operation and optimize its prediction.
        # only feasible if prediction heads are verb/manipulated/affected. 
        self.with_operation_pred = with_op_pred
        if self.with_operation_pred:
            fuse_mat_path = "data/annotations/fuse_matrix.npy"
            fuse_weights = [1.0/3, 1.0/3, 1.0/3]
            fuse_mat = torch.from_numpy(np.load(fuse_mat_path)).to(torch.float32)
            assert len(self.num_classes) == 3, "3 entities are needed to predict operations."
            assert fuse_mat.shape[1] == sum(self.num_classes), "Prediction heads should be 'verb', 'manipulated', and 'affected'."
            fuse_mat[fuse_mat.to(torch.bool)] = torch.tensor(fuse_weights).repeat(fuse_mat.shape[0])
            self.op_pred_method = op_pred_method  # cls_head or fuse
            if self.op_pred_method == "fuse":
                # calculate operation probability by multiplying corresponding entity probabilities.
                # no training is needed.
                self.prediction_fuser = PredictionFuser(fuse_groups=[[0, 1, 2]], fuse_weight_mats=[fuse_mat.cuda()])
            elif self.op_pred_method == "cls_head":
                # get operation probability by prediting using a convolution head with concatenated entitiy probabilities as input.
                # train cls_head with cross-entropy.
                self.op_cls_head = PtTransformerClsHead(
                                    fuse_mat.shape[1], head_dim, fuse_mat.shape[0],
                                    kernel_size=head_kernel_size,
                                    prior_prob=self.train_cls_prior_prob,
                                    with_ln=head_with_ln,
                                    num_layers=head_num_layers
                                )
            else:
                ValueError("Invalid operation pred_method")
            
            # op2comb: [#op, 3] (verb idx, manipulaed idx, affected idx) for each operation
            op2comb = torch.nonzero(fuse_mat)[:, 1].view(-1, 3)
            op2comb -= torch.tensor([0, self.num_classes[0], self.num_classes[0] + self.num_classes[1]]).unsqueeze(0)
            self.comb2op = dict(zip([tuple(x) for x in op2comb.tolist()], range(len(op2comb))))
            # update num_classes and num_cls_types
            self.num_cls_types += 1
            self.num_classes.append(len(op2comb))
            
            # whether apply graph modeling　(=grpe) for operation.
            self.apply_op_graph_modeling = apply_op_graph_modeling
            if self.apply_op_graph_modeling:
                self.op_grpe_adj_fr_num = 10
                # GRPE architecture config
                self.op_grpe_input_dim = len(op2comb)
                self.op_grpe_hidden_dim = 200
                self.op_grpe_num_layers = 2
                self.op_grpe_num_heads = 4
                self.op_grpe_dropout = 0.
                self.op_grpe_att_dropout = 0.
                self.op_grpe_num_edge_type = 1  # not necessary
                self.op_grpe_max_hop = self.op_grpe_adj_fr_num * 2
                self.op_grpe_use_independent_token = False
                self.op_grpe_perturb_noise = 0.
                self.op_grpe_num_last_mlp = 0
                
                self.op_grpe_net = GRPENetwork(self.op_grpe_input_dim, self.op_grpe_hidden_dim, self.op_grpe_input_dim,
                                            num_layer=self.op_grpe_num_layers, nhead=self.op_grpe_num_heads,
                                            dropout=self.op_grpe_dropout, attention_dropout=self.op_grpe_att_dropout,
                                            max_hop=self.op_grpe_max_hop, num_edge_type=self.op_grpe_num_edge_type, 
                                            use_independent_token=self.op_grpe_use_independent_token,
                                            perturb_noise=self.op_grpe_perturb_noise, num_last_mlp=self.op_grpe_num_last_mlp,
                                            )
                self.refine_op_cls_head = PtTransformerClsHead(
                                    self.op_grpe_input_dim,
                                    head_dim, len(op2comb),
                                    kernel_size=head_kernel_size,
                                    prior_prob=self.train_cls_prior_prob,
                                    with_ln=head_with_ln,
                                    num_layers=head_num_layers
                                )
        
        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):        
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        # out_cls: #types of List[B, #cls + 1, T_i]
        out_cls_logits = [self.cls_head[i](fpn_feats, fpn_masks) for i in range(len(self.cls_head))]
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: N List[F List[B, C, T_i]] -> N List[F List[B, T_i, C]]
        out_cls_logits = [[x.permute(0, 2, 1) for x in out_cls_logits[i]] for i in range(len(self.cls_head))]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]
        
        if self.with_operation_pred:
            # add operation prediction to out_cls_logits
            out_cls_logits = self.add_operation_prediction(video_list, points, out_cls_logits, out_offsets, fpn_masks)
            if self.apply_op_graph_modeling:
                # apply graph modeling for operation predictions.
                init_op_cls_logits = out_cls_logits[-1]
                op_refined_cls_logits = self.post_op_grpe(video_list, points, out_offsets, init_op_cls_logits, fpn_feats, fpn_masks)
                op_refined_cls_logits = [x.permute(0, 2, 1) for x in op_refined_cls_logits]
                
        # return loss during training
        if self.training:
            # generate segment/label List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [[x['labels'][i].to(self.device) for i in range(len(self.cls_head))] for x in video_list]
            if self.with_operation_pred:
                # add operation gt.
                gt_op_labels = [[self.comb2op[tuple(comb)] for comb in zip(x[0].tolist(), x[1].tolist(), x[2].tolist())] for x in gt_labels]
                gt_labels = [x + [torch.tensor(gt_op_labels[i]).to(self.device)] for i, x in enumerate(gt_labels)]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            # compute the loss and return
            loss_names = ["verb_loss", "manipulated_loss", "affected_loss", "operation_loss"] \
                if self.with_operation_pred and self.op_pred_method == "cls_head" \
                else ["verb_loss", "manipulated_loss", "affected_loss"]
            if self.with_operation_pred and self.op_pred_method == "fuse":
                # Fuser donesn't need training, so just train entity logits.
                # Note that fuser applies sigmoid to entity logits.
                losses = self.losses(
                    fpn_masks,
                    out_cls_logits[:-1], out_offsets,
                    gt_cls_labels[:-1], gt_offsets,
                    is_sigmoid_done=True,
                    loss_names=loss_names
                )
            else:
                losses = self.losses(
                    fpn_masks,
                    out_cls_logits, out_offsets,
                    gt_cls_labels, gt_offsets,
                    is_sigmoid_done=False,
                    loss_names=loss_names
                )
            
            if self.with_operation_pred and self.apply_op_graph_modeling:
                # calculate loss for refined operation logits.
                refined_losses = self.losses(
                    fpn_masks, 
                    [op_refined_cls_logits], out_offsets,
                    [gt_cls_labels[-1]], gt_offsets
                )
                losses['refined_operation_loss'] = refined_losses['cls_loss']
                losses['final_loss'] = losses['final_loss'] + refined_losses['cls_loss']
            
            return losses
        else:
            # !!!!return logits after sigmoid!!!!
            if (not self.with_operation_pred) or self.op_pred_method != "fuse":
                out_cls_logits = [[x.sigmoid() for x in type_cls_logits] for type_cls_logits in out_cls_logits]
            if self.with_operation_pred and self.apply_op_graph_modeling:
                out_cls_logits[-1] = [x.sigmoid() for x in op_refined_cls_logits]
            return points, fpn_masks, out_cls_logits, out_offsets

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [[] for _ in range(self.num_cls_types)], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            # cls_targets > [(F T x C0), (F T x C1), ...]
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            for i in range(self.num_cls_types):
                gt_cls[i].append(cls_targets[i])
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x #types
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = [gt_segment.new_full((num_pts, self.num_classes[i]), 0) for i in range(self.num_cls_types)]
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: #types × F T x Ci; reg_targets F T x 2
        cls_targets = []
        for i in range(self.num_cls_types):
            gt_label_one_hot = F.one_hot(
                gt_label[i], self.num_classes[i]
            ).to(reg_targets.dtype)
            cls_targets.append(min_len_mask @ gt_label_one_hot)
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets[-1].clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, 
        is_sigmoid_done=False, loss_names=[]
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = []
        for i in range(len(gt_cls_labels)):
            gt_cls.append(torch.stack(gt_cls_labels[i]))
        pos_mask = torch.logical_and((gt_cls[0].sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # verb, manipulated, affected, (operation)
        sum_cls_loss = torch.tensor(0., dtype=torch.float32).to(self.device)
        cls_losses = []
        for i in range(len(gt_cls)):
            # gt_cls is already one hot encoded now, simply masking out
            gt_target = gt_cls[i][valid_mask]

            # optinal label smoothing
            gt_target *= 1 - self.train_label_smoothing
            gt_target += self.train_label_smoothing / (self.num_classes[i] + 1)

            # focal loss
            cls_loss = sigmoid_focal_loss(
                torch.cat(out_cls_logits[i], dim=1)[valid_mask],
                gt_target,
                reduction='sum',
                is_sigmoid_done=is_sigmoid_done
            )
            cls_loss /= self.loss_normalizer
            sum_cls_loss += cls_loss
            cls_losses.append(cls_loss)

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = sum_cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = sum_cls_loss + reg_loss * loss_weight
        loss_dict = {'cls_loss'   : sum_cls_loss,
                    'reg_loss'   : reg_loss,
                    'final_loss' : final_loss}
        if len(loss_names):
            assert len(loss_names) == len(gt_cls_labels) == len(out_cls_logits)
            for i, loss_name in enumerate(loss_names):
                loss_dict[loss_name] = cls_losses[i]
        return loss_dict
        
    def add_operation_prediction(self, video_list, points, out_cls_logits, out_offsets, fpn_masks):
        # out_*: N List[F List[B, T_i, C]] (cls_logits for verb, manipulated, affected)
        assert self.with_operation_pred
        assert len(out_cls_logits) == 3
        ### prediction fuserを使った掛け算 
        ### !!結果は全てsigmoid済みになることに注意
        if self.op_pred_method == "fuse":
            self.prediction_fuser.reset()
            for i in range(len(out_cls_logits)):
                self.prediction_fuser.add_logits(out_cls_logits[i])
                self.prediction_fuser.add_offsets(out_offsets)
            out_cls_logits, _ = self.prediction_fuser.fuse()
        ### 連結してhead予測
        ### !!結果はsigmoid済みにならない
        elif self.op_pred_method == "cls_head":
            level_splits = [len(x) for x in points]
            ent_cat_cls_logits = torch.cat([torch.cat(x, 1).permute(0, 2, 1) for x in out_cls_logits], 1)  # [B, C, T]
            op_cls_logits = self.op_cls_head(torch.split(ent_cat_cls_logits, level_splits, dim=-1), [x.unsqueeze(1) for x in fpn_masks])  
            out_cls_logits.append([x.permute(0, 2, 1) for x in op_cls_logits])
        return out_cls_logits  # F List[B, T_i, C]
             
    def post_op_grpe(self, video_list, points, init_offsets, init_cls_logits, fpn_feats, fpn_masks):   
        level_splits = [x.shape[2] for x in fpn_feats]   
        # F List[B, T_i, C] -> [B, FT, C]
        fpn_feats = [x.permute(0, 2, 1) for x in fpn_feats]
        fpn_feats = torch.cat(fpn_feats, dim=1)
        valid_masks = torch.cat(fpn_masks, dim=1)
        init_offsets = torch.cat(init_offsets, dim=1).detach().clone()
        # calculate segment
        num_vid = init_offsets.shape[0]
        num_pts = init_offsets.shape[1]
        points_cat = torch.cat(points).detach().unsqueeze(0).repeat(num_vid, 1, 1)
        init_seg_left = points_cat[:, :, 0] - init_offsets[:, :, 0] * points_cat[:, :, 3]
        init_seg_right = points_cat[:, :, 0] + init_offsets[:, :, 1] * points_cat[:, :, 3]
        # apply graph processing per video
        init_op_logits = torch.cat(init_cls_logits, dim=1)
        refined_feats = []
        for i in range(num_vid):
            # sort by time
            temp_sort_idxs = torch.from_numpy(np.lexsort((init_seg_left[i].cpu().numpy(), init_seg_right[i].cpu().numpy()))).to(points_cat.device)
            valid_mask_per_vid = valid_masks[i][temp_sort_idxs]
            # extract only valid operation segments and sampling by score
            valid_idxs = temp_sort_idxs[valid_mask_per_vid]
            
            # split operation into entities 
            op_logits = init_op_logits[i, valid_idxs]
            num_nodes = len(op_logits)
            
            # create distance matrix
            distance_mat = torch.zeros((1, num_nodes, num_nodes), device=self.device, dtype=torch.int64)
            seg_dist_mat = (torch.arange(num_nodes).unsqueeze(0).repeat(num_nodes, 1) - torch.arange(num_nodes).unsqueeze(-1).repeat(1, num_nodes)).to(self.device)
            intra_seg_mask = (seg_dist_mat == 0)
            inter_seg_mask = ~intra_seg_mask
            distance_mat[0] += torch.tril(seg_dist_mat + self.op_grpe_adj_fr_num + 1, diagonal=-1) * inter_seg_mask  # pre-segment
            distance_mat[0] += torch.triu(seg_dist_mat + self.op_grpe_adj_fr_num, diagonal=1) * inter_seg_mask  # post-segment
            distance_mat[(distance_mat > self.op_grpe_max_hop) | (distance_mat < 0)] = -1
            # for e in range(6):
            #     print(','.join(map(str, distance_mat[0][e].tolist()[:50])))
            
            # create edge attr matrix
            # 0: self, 1: connected
            edge_attr_mat = -torch.ones((1, num_nodes, num_nodes), device=self.device, dtype=torch.int64)
            edge_attr_mat[distance_mat != -1] = 1
            edge_attr_mat[0, torch.arange(num_nodes), torch.arange(num_nodes)] = 0
            # for e in range(6):
            #     print(','.join(map(str, edge_attr_mat[0][e].tolist()[:50])))

            # GRPE
            op_feats = self.op_grpe_net(
                    op_logits.unsqueeze(0),
                    mask=torch.zeros((1, num_nodes), device=op_logits.device, dtype=torch.bool),
                    distance_mat=distance_mat,
                    edge_attr_mat=edge_attr_mat)[0]
            pad_cls_feats = init_op_logits[i].detach().clone()
            pad_cls_feats[valid_idxs] = op_feats
            refined_feats.append(torch.split(pad_cls_feats, level_splits))
        # classification / regression as refinement
        # convert dim order (B List[F T_i C] -> F List [B C T_i]) to fit in cls_head / reg_head
        convert_refined_feats = []
        for i in range(len(level_splits)):
            refined_feats_i = [x[i].T for x in refined_feats]
            refined_feats_i = torch.stack(refined_feats_i)
            convert_refined_feats.append(refined_feats_i)
        # pass converted feats to cls_head / reg_head
        refined_cls_logits = self.refine_op_cls_head(convert_refined_feats, [x.unsqueeze(1) for x in fpn_masks])
        return refined_cls_logits  # F List[B, C, T_i]
