import torch
from torch import nn
import torch.nn.functional as F


class extension_layer(nn.Module):
    def __init__(self, input_dim, use_manipulatedstate=False, predict_affectedobj=False, use_affectedstate=False):
        super(extension_layer, self).__init__()
        self.use_manipulatedstate = use_manipulatedstate
        self.predict_affectedobj = predict_affectedobj
        self.use_affectedstate = use_affectedstate
        self.init_layers_weights(input_dim)

    def forward(self, input):
        if (len(input.shape)) == 2:
            input = input.unsqueeze(0)
        
        out = {}
        out["hand_states"] = self.hand_state_layer(input)
        dxdymagnitude_pred = self.hand_dxdy_layer(input)
        dxdymagnitude_pred_sub = 0.1 * F.normalize(dxdymagnitude_pred[:,:,1:], p=2, dim=2)
        out["hand_dxdymagnitudes"] = torch.cat([dxdymagnitude_pred[:,:,0].unsqueeze(-1), dxdymagnitude_pred_sub], dim=2)
        
        if self.use_manipulatedstate:
            out["manipulated_states"] = self.manipulated_state_layer(input)
        if self.predict_affectedobj:
            out["affecting_states"] = self.affecting_state_layer(input)
            dxdymagnitude_pred = self.manip_dxdy_layer(input)
            dxdymagnitude_pred_sub = 0.1 * F.normalize(dxdymagnitude_pred[:,:,1:], p=2, dim=2)
            out["manip_dxdymagnitudes"] = torch.cat([dxdymagnitude_pred[:,:,0].unsqueeze(-1), dxdymagnitude_pred_sub], dim=2)
        if self.use_affectedstate:
            out["affected_states"] = self.affected_state_layer(input)
        return out

    def init_layers_weights(self, input_dim):
        self.hand_state_layer = nn.Sequential(nn.Linear(input_dim, 32), \
            nn.ReLU(), \
            nn.Dropout(p=0.5),\
            nn.Linear(32, 2)) # non-manipulating or manipulating
        self.hand_dxdy_layer = torch.nn.Linear(input_dim, 3)  # (magnitude, dx, dy)
        
        if self.use_manipulatedstate:
            self.manipulated_state_layer = nn.Sequential(nn.Linear(input_dim, 32), \
                nn.ReLU(), \
                nn.Dropout(p=0.5),\
                nn.Linear(32, 2)) # non-manipulated or manipulated
        if self.predict_affectedobj:
            self.affecting_state_layer = nn.Sequential(nn.Linear(input_dim, 32), \
                nn.ReLU(), \
                nn.Dropout(p=0.5),\
                nn.Linear(32, 2)) # non-affecting or affecting
            self.manip_dxdy_layer = torch.nn.Linear(input_dim, 3)  # (magnitude, dx, dy)
        if self.use_affectedstate:
            self.affected_state_layer = nn.Sequential(nn.Linear(input_dim, 32), \
                nn.ReLU(), \
                nn.Dropout(p=0.5),\
                nn.Linear(32, 2)) # non-affected or affected
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) 
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.hand_state_layer[0], 0, 0.01)
        normal_init(self.hand_state_layer[3], 0, 0.01)
        normal_init(self.hand_dxdy_layer, 0, 0.01)
        if self.use_manipulatedstate:
            normal_init(self.manipulated_state_layer[0], 0, 0.01)
            normal_init(self.manipulated_state_layer[3], 0, 0.01)
        if self.predict_affectedobj:
            normal_init(self.affecting_state_layer[0], 0, 0.01)
            normal_init(self.affecting_state_layer[3], 0, 0.01)
            normal_init(self.manip_dxdy_layer, 0, 0.01)
        if self.use_affectedstate:
            normal_init(self.affected_state_layer[0], 0, 0.01)
            normal_init(self.affected_state_layer[3], 0, 0.01)
