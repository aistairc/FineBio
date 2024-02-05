# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn


class GRPENetwork(nn.Module):
    def __init__(
        self,
        input_node_dim,
        hidden_dim,
        output_dim,
        num_layer=6,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        max_hop=256,
        num_edge_type=25,
        use_independent_token=False,
        perturb_noise=0.0,
        num_last_mlp=0
    ):
        super().__init__()
        self.perturb_noise = perturb_noise
        self.max_hop = max_hop
        self.num_edge_type = num_edge_type
        self.use_independent_token = use_independent_token

        self.node_emb = nn.Linear(input_node_dim, hidden_dim)

        self.UNREACHABLE_DISTANCE = max_hop + 1
        
        self.NO_EDGE = num_edge_type + 1
                
        if not self.use_independent_token:
            self.query_hop_emb = nn.Embedding(max_hop + 2, hidden_dim)
            self.query_edge_emb = nn.Embedding(num_edge_type + 2,  hidden_dim)
            self.key_hop_emb = nn.Embedding(max_hop + 2, hidden_dim)
            self.key_edge_emb = nn.Embedding(num_edge_type + 2, hidden_dim)
            self.value_hop_emb = nn.Embedding(max_hop + 2, hidden_dim)
            self.value_edge_emb = nn.Embedding(num_edge_type + 2, hidden_dim)
        else:
            self.query_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 2, hidden_dim) for _ in range(num_layer)]
            )
            self.query_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 2, hidden_dim) for _ in range(num_layer)]
            )
            self.key_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 2, hidden_dim) for _ in range(num_layer)]
            )
            self.key_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 2, hidden_dim) for _ in range(num_layer)]
            )
            self.value_hop_emb = nn.ModuleList(
                [nn.Embedding(max_hop + 2, hidden_dim) for _ in range(num_layer)]
            )
            self.value_edge_emb = nn.ModuleList(
                [nn.Embedding(num_edge_type + 2, hidden_dim) for _ in range(num_layer)]
            )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_dim,
                    ffn_size=dim_feedforward,
                    dropout_rate=dropout,
                    attention_dropout_rate=attention_dropout,
                    num_heads=nhead,
                )
                for _ in range(num_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.last_mlp = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_last_mlp)
            ]
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def encode_node(self, x):
        return self.node_emb(x)

    def forward(self, x, mask, distance_mat, edge_attr_mat):
        """
        Args
            x (torch.tensor)           : node feature [B, N, D]
            mask (torch.tensor)        : boolean mask (1 for invalid nodes). [B, N]
            distance_mat (torch.tensor): distance matrix. [B, N, N]
        """
            
        x = self.encode_node(x)

        if self.training:
            perturb = torch.empty_like(x).uniform_(
                -self.perturb_noise, self.perturb_noise
            )
            x = x + perturb

        # node features
        input_x = x

        # invalid node mask
        input_mask = mask

        # input distance mask
        distance_mat = distance_mat.clamp(
            max=self.max_hop
        )
        input_distance_mask = distance_mat
        input_distance_mask[input_distance_mask == -1] = self.UNREACHABLE_DISTANCE
        
        # input edge type mat
        edge_attr_mat = edge_attr_mat.clamp(
            max=self.num_edge_type
        )
        input_edge_attr_mat = edge_attr_mat
        input_edge_attr_mat[input_edge_attr_mat == -1] = self.NO_EDGE

        for i, enc_layer in enumerate(self.layers):
            if self.use_independent_token:
                output = enc_layer(
                    input_x,
                    self.query_hop_emb[i].weight,
                    self.query_edge_emb[i].weight,
                    self.key_hop_emb[i].weight,
                    self.key_edge_emb[i].weight,
                    self.value_hop_emb[i].weight,
                    self.value_edge_emb[i].weight,
                    input_distance_mask,
                    input_edge_attr_mat,
                    mask=input_mask
                )
            else:
                output = enc_layer(
                    input_x,
                    self.query_hop_emb.weight,
                    self.query_edge_emb.weight,
                    self.key_hop_emb.weight,
                    self.key_edge_emb.weight,
                    self.value_hop_emb.weight,
                    self.value_edge_emb.weight,
                    input_distance_mask,
                    input_edge_attr_mat,
                    mask=input_mask
                )

        output = self.final_ln(output)
        output = self.last_mlp(output)
        output = self.linear(output)

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb, 
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [B, H, N, d_k]
        v = v.transpose(1, 2)  # [B, H, N, d_v]
        k = k.transpose(1, 2)  # [B, H, N, d_k]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)  # [1, H, num_hop_types, d_k]
        query_edge_emb = query_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)  # [1, H, num_edge_types, d_k]
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)  # [1, H, num_hop_types, d_k]
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)  # [1, H, num_edge_types, d_k]
        

        # [B, H, N, num_hop_types]
        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        # [B, H, N, num_edge_types]
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        

        # [B, H, N, #hop_types]
        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop.transpose(2, 3), 2, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )
        # [B, H, N, num_edge_types]
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        spatial_bias = query_hop + key_hop
        edge_bias = query_edge + key_edge

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bias
        x = x * self.scale

        if mask is not None:
            x = x.masked_fill(
                mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf")
            )

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)

        value_hop_emb = value_hop_emb.view(
            1, num_hop_types, self.num_heads, self.att_size
        ).transpose(1, 2)
        value_edge_emb = value_edge_emb.view(
            1, num_edge_types, self.num_heads, self.att_size
        ).transpose(1, 2)

        # [B, H, N, num_hop_types]
        value_hop_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_hop_types),
            device=value_hop_emb.device,
        )
        value_hop_att = torch.scatter_add(
            value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x
        )
        # [B, H, N, num_edge_types]
        value_edge_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_edge_types),
            device=value_hop_emb.device,
        )
        value_edge_att = torch.scatter_add(
            value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x
        )
        

        x = torch.matmul(x, v) + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size,
            attention_dropout_rate,
            num_heads,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None
    ):
        y = self.self_attention_norm(x)
        y = self.self_attention(
            y,
            y,
            y,
            query_hop_emb,
            query_edge_emb,
            key_hop_emb,
            key_edge_emb,
            value_hop_emb,
            value_edge_emb,
            distance,
            edge_attr,
            mask=mask
        )
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
