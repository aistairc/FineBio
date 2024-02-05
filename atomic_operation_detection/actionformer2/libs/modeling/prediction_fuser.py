import torch


class PredictionFuser():
    """
    Fuse predictions for different attributes.
    """
    def __init__(self, fuse_groups, fuse_weight_mats):
        """
        Args:
            test_cfg: cfg for test
            num_classes: #classes for each model head
            fuse_groups: groups of model head indexes to be integrated into a new single label.
            fuse_weight_mat: matrix to fuse different types of labels into fused-label (# of fused label, sum(# of souce label))
            score_by: which type NMS is done by (only used for inference_together)
        """
        self.list_out_class_logits = []
        self.list_out_offsets = []

        # when fuse_groups exists, calculate fused-label
        self.fuse_groups = fuse_groups
        self.fuse_weight_mats = fuse_weight_mats
    
    def add_logits(self, out_class_logits):
        self.list_out_class_logits.append(out_class_logits)
    
    def add_offsets(self, out_offsets):
        self.list_out_offsets.append(out_offsets)
        
    def calc_cls_score(self):
        # list_out_*: N (List) [F (List) [B, T_i, C]]
        for type_num in range(len(self.list_out_class_logits)):  
            for level in range(len(self.list_out_class_logits[type_num])):
                self.list_out_class_logits[type_num][level] = self.list_out_class_logits[type_num][level].sigmoid()
    
    def fuse_labels(self):
        for fuse_group, fuse_weight_mat in zip(self.fuse_groups, self.fuse_weight_mats):
            # N (list)[F (list)[B, T_i, C]] -> F (list)[N (list)[B, T_i, C]]
            fused_logits = [[self.list_out_class_logits[type_num][level] for type_num in fuse_group] for level in range(len(self.list_out_class_logits[0]))]
            # F (list)[N (list)[B, T_i, C]] -> F (list)[B, T_i, sum(C)]
            fused_logits = [torch.concat(x, dim=-1) for x in fused_logits]
            # multiply: (p(a) ** weight[0]) * (p(b) ** weight[1]) * ...
            # F (list)[B, T_i, C]
            fused_logits = [torch.prod(x.unsqueeze(-2) ** fuse_weight_mat, -1) for x in fused_logits]
            self.list_out_class_logits.append(fused_logits)

    def fuse_offsets(self):
        for fuse_group, fuse_weight_mat in zip(self.fuse_groups, self.fuse_weight_mats):
            fuse_weights = fuse_weight_mat[0][fuse_weight_mat[0] > 0]
            # N (list)[F (list)[B, T_i, C]] -> F (list)[N (list)[B, T_i, 1, C]]
            fused_offsets = [[self.list_out_offsets[type_num][level].unsqueeze(2) for type_num in fuse_group] for level in range(len(self.list_out_offsets[0]))]
            # F (list)[N (list)[B, T_i, 1, C]] -> F (list)[B, T_i, N, C]
            fused_offsets = [torch.concat(x, dim=2) for x in fused_offsets]
            # F (list)[B, T_i, N, C] -> F (list)[B, T_i, C]
            fused_offsets = [torch.sum(x * fuse_weights.unsqueeze(-1), dim=2) for x in fused_offsets]
            self.list_out_offsets.append(fused_offsets)

    def fuse(self, is_sigmoid_done=False):
        # 0: Convert logits into score before fusion if not converted yet
        if not is_sigmoid_done:
            self.calc_cls_score()
        # 1: integrate labels and calculate logits for fused-labels
        self.fuse_labels()
        self.fuse_offsets()
        return self.list_out_class_logits, self.list_out_offsets
        
    def reset(self):
        self.list_out_class_logits = []
        self.list_out_offsets = []