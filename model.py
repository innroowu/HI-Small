import dgl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import torch
from torch_geometric.nn import APPNP, EdgeConv, LEConv,TransformerConv,GCNConv, SGConv, SAGEConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import *

class CrossAttention(nn.Module):
    """Cross-Attention module for in/out sequences"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # 🔧 添加權重初始化
        self._reset_parameters()
        
    
    def _reset_parameters(self):
        """初始化權重"""
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
        if self.q_linear.bias is not None:
            nn.init.constant_(self.q_linear.bias, 0)
            nn.init.constant_(self.k_linear.bias, 0)
            nn.init.constant_(self.v_linear.bias, 0)
            nn.init.constant_(self.out_linear.bias, 0)

        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # 🔧 修復：確保 mask 至少有一個 True
            # 如果整行都是 False，至少保留第一個位置
            mask_sum = mask.sum(dim=-1, keepdim=True)
            mask_fixed = mask.clone()
            mask_fixed[:, :, 0] = torch.where(mask_sum.squeeze(-1) == 0, True, mask_fixed[:, :, 0])
            
            scores = scores.masked_fill(mask_fixed.unsqueeze(1) == 0, float('-1e9'))  # 改用 -1e9 而非 -inf
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # 🔧 修復：檢查並替換 NaN
        if torch.isnan(attention_weights).any():
            attention_weights = torch.where(
                torch.isnan(attention_weights),
                torch.zeros_like(attention_weights),
                attention_weights
            )
        
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_linear(output)
        
        return output, attention_weights


class Binary_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rnn_in_channels, encoder_layer='gcn', decoder='mlp', rnn='gru', rnn_agg='last', num_layers=2,
                 decoder_layers=1, dropout=0.5, bias=True, save_mem=True, use_bn=True, concat_feature=1, emb_first=1, heads=1, lstm_norm='ln', gnn_norm='bn', graph_op='', aggr='add',
                 use_cross_attention=False, cross_attn_heads=4):  # 新增參數
        super(Binary_Classifier, self).__init__()
        self.rnn_agg = rnn_agg
        self.rnn = rnn
        self.concat_feature = concat_feature
        self.emb_first = emb_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_layers = encoder_layers = num_layers
        self.decoder_layers = decoder_layers
        self.lstm_norm = lstm_norm
        self.gnn_norm = gnn_norm
        self.graph_op = graph_op
        self.use_cross_attention = use_cross_attention  # 新增
        rnn_out_channels = int(hidden_channels/2)
        
        # Initialize LSTM part
        if emb_first:
            self.lstm_emb_in = nn.Linear(rnn_in_channels, rnn_out_channels)
            self.lstm_emb_out = nn.Linear(rnn_in_channels, rnn_out_channels)
            if self.lstm_norm == 'bn':
                self.lstm_emb_norm_in = nn.BatchNorm1d(rnn_out_channels)
                self.lstm_emb_norm_out = nn.BatchNorm1d(rnn_out_channels)
            elif self.lstm_norm == 'ln':
                self.lstm_emb_norm_in = nn.LayerNorm(rnn_out_channels)
                self.lstm_emb_norm_out = nn.LayerNorm(rnn_out_channels)
            if rnn == 'lstm':
                self.lstm_in = nn.LSTM(rnn_out_channels, rnn_out_channels)
                self.lstm_out = nn.LSTM(rnn_out_channels, rnn_out_channels)
            elif rnn == 'gru':
                self.lstm_in = nn.GRU(rnn_out_channels, rnn_out_channels)
                self.lstm_out = nn.GRU(rnn_out_channels, rnn_out_channels)
        else:
            if rnn == 'lstm':
                self.lstm_in = nn.LSTM(rnn_in_channels, rnn_out_channels)
                self.lstm_out = nn.LSTM(rnn_in_channels, rnn_out_channels)
            elif rnn == 'gru':
                self.lstm_in = nn.GRU(rnn_in_channels, rnn_out_channels)
                self.lstm_out = nn.GRU(rnn_in_channels, rnn_out_channels)
        
        # Cross-Attention modules (新增)
        if self.use_cross_attention:
            print("Using cross attention!!!!")
            self.cross_attn_in2out = CrossAttention(rnn_out_channels, num_heads=cross_attn_heads, dropout=dropout)
            self.cross_attn_out2in = CrossAttention(rnn_out_channels, num_heads=cross_attn_heads, dropout=dropout)
            
        # Initialize GNN part
        self.encoder = nn.ModuleList()
        self.encoder_layer = encoder_layer
        use_rnn = 1
        if 'dualcata' in encoder_layer:
            atten_hidden = encoder_layer.split('-')[-1]
            if atten_hidden.isdigit():
                atten_hidden = int(atten_hidden)
            else:
                atten_hidden = 16
                
            self.encoder.append(
                DualCATAConv(hidden_channels, hidden_channels, bias=bias, atten_hidden=atten_hidden, aggr=aggr))
            for _ in range(encoder_layers-1):
                self.encoder.append(
                    DualCATAConv(hidden_channels, hidden_channels, bias=bias, atten_hidden=atten_hidden, aggr=aggr))
        else:
            raise NameError(f'{encoder_layer} is not implemented!')
        
        # Initialize decoder
        self.decoder = nn.ModuleList()
        for _ in range(decoder_layers-1):
            self.decoder.append(nn.Linear(hidden_channels, hidden_channels))
        self.decoder.append(nn.Linear(hidden_channels, out_channels))
        
        # Initialize other modules
        self.dropout = dropout
        self.activation = F.relu
        # Normalization layer after each encoder layer
        self.bns = nn.ModuleList()
        if self.lstm_norm == 'bn':
            self.lstm_norm_in = nn.BatchNorm1d(rnn_out_channels)
            self.lstm_norm_out = nn.BatchNorm1d(rnn_out_channels)
        elif self.lstm_norm == 'ln':
            self.lstm_norm_in = nn.LayerNorm(rnn_out_channels)
            self.lstm_norm_out = nn.LayerNorm(rnn_out_channels)
        if self.gnn_norm == 'ln':
            for _ in range(self.encoder_layers):
                self.bns.append(nn.LayerNorm(hidden_channels))
        elif self.gnn_norm == 'bn':
            for _ in range(self.encoder_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
    def forward(self, in_pack, out_pack, lens_in, lens_out, edge_index=None, edge_attr=None): 
        # generate lstm embeddings
        if self.emb_first:
            in_pack = self.lstm_emb_in(in_pack)
            in_pack = self.lstm_emb_norm_in(in_pack)
            out_pack = self.lstm_emb_out(out_pack)
            out_pack = self.lstm_emb_norm_out(out_pack)
            in_pack = pack_padded_sequence(in_pack, lens_in.cpu(), batch_first=True, enforce_sorted=False)
            out_pack = pack_padded_sequence(out_pack, lens_out.cpu(), batch_first=True, enforce_sorted=False)
        
        # RNN encoding
        if self.rnn == 'lstm':
            edges_in, (h_in, c_in) = self.lstm_in(in_pack)
            edges_out, (h_out, c_out) = self.lstm_out(out_pack)
        elif self.rnn == 'gru':
            edges_in, h_in = self.lstm_in(in_pack)
            edges_out, h_out = self.lstm_out(out_pack)
        
        # Unpack sequences for cross-attention or pooling
        edges_in_unpacked, lens_in_unpacked = pad_packed_sequence(edges_in, batch_first=True)
        edges_out_unpacked, lens_out_unpacked = pad_packed_sequence(edges_out, batch_first=True)
        
        if self.use_cross_attention:
            # 統一設備
            device = edges_in_unpacked.device
            lens_in_unpacked = lens_in_unpacked.to(device)
            lens_out_unpacked = lens_out_unpacked.to(device)
            
            # Create padding masks
            batch_size = edges_in_unpacked.size(0)
            max_len_in = edges_in_unpacked.size(1)
            max_len_out = edges_out_unpacked.size(1)
            
            # 🔧 修復：確保 lens 至少為 1（避免除以零）
            lens_in_unpacked = torch.clamp(lens_in_unpacked, min=1)
            lens_out_unpacked = torch.clamp(lens_out_unpacked, min=1)
            
            # Create masks
            mask_in = torch.arange(max_len_in, device=device).unsqueeze(0) < lens_in_unpacked.unsqueeze(1)
            mask_out = torch.arange(max_len_out, device=device).unsqueeze(0) < lens_out_unpacked.unsqueeze(1)
            
            # 🔧 修復：確保至少有一個 True
            # 如果 mask 全為 False，設置第一個位置為 True
            mask_in[:, 0] = True
            mask_out[:, 0] = True
            
            # Cross-attention masks
            mask_in2out = mask_in.unsqueeze(2) * mask_out.unsqueeze(1)
            mask_out2in = mask_out.unsqueeze(2) * mask_in.unsqueeze(1)
            
            edges_in_attn, _ = self.cross_attn_in2out(edges_in_unpacked, edges_out_unpacked, edges_out_unpacked, mask_in2out)
            edges_out_attn, _ = self.cross_attn_out2in(edges_out_unpacked, edges_in_unpacked, edges_in_unpacked, mask_out2in)
            
            # 🔧 修復：檢查並替換 NaN
            if torch.isnan(edges_in_attn).any():
                edges_in_attn = torch.where(
                    torch.isnan(edges_in_attn),
                    torch.zeros_like(edges_in_attn),
                    edges_in_attn
                )
            if torch.isnan(edges_out_attn).any():
                edges_out_attn = torch.where(
                    torch.isnan(edges_out_attn),
                    torch.zeros_like(edges_out_attn),
                    edges_out_attn
                )
            
            # Pooling after cross-attention
            if self.rnn_agg == 'last':
                # 🔧 修復：確保索引有效
                valid_idx_in = torch.clamp(lens_in_unpacked - 1, min=0, max=max_len_in - 1)
                valid_idx_out = torch.clamp(lens_out_unpacked - 1, min=0, max=max_len_out - 1)
                h_in = edges_in_attn[torch.arange(batch_size, device=device), valid_idx_in]
                h_out = edges_out_attn[torch.arange(batch_size, device=device), valid_idx_out]
            elif self.rnn_agg == 'max':
                edges_in_attn = edges_in_attn.masked_fill(~mask_in.unsqueeze(-1), float('-1e9'))  # 改用 -1e9
                edges_out_attn = edges_out_attn.masked_fill(~mask_out.unsqueeze(-1), float('-1e9'))
                h_in = torch.max(edges_in_attn, dim=1)[0]
                h_out = torch.max(edges_out_attn, dim=1)[0]
            elif self.rnn_agg == 'mean':
                edges_in_attn = edges_in_attn.masked_fill(~mask_in.unsqueeze(-1), 0)
                edges_out_attn = edges_out_attn.masked_fill(~mask_out.unsqueeze(-1), 0)
                # 🔧 修復：避免除以零
                sum_in = edges_in_attn.sum(dim=1)
                sum_out = edges_out_attn.sum(dim=1)
                count_in = lens_in_unpacked.unsqueeze(-1).float().clamp(min=1)
                count_out = lens_out_unpacked.unsqueeze(-1).float().clamp(min=1)
                h_in = sum_in / count_in
                h_out = sum_out / count_out
            elif self.rnn_agg == 'sum':
                edges_in_attn = edges_in_attn.masked_fill(~mask_in.unsqueeze(-1), 0)
                edges_out_attn = edges_out_attn.masked_fill(~mask_out.unsqueeze(-1), 0)
                h_in = edges_in_attn.sum(dim=1)
                h_out = edges_out_attn.sum(dim=1)
            
            # 🔧 修復：最終檢查 NaN
            if torch.isnan(h_in).any():
                h_in = torch.where(torch.isnan(h_in), torch.zeros_like(h_in), h_in)
            if torch.isnan(h_out).any():
                h_out = torch.where(torch.isnan(h_out), torch.zeros_like(h_out), h_out)
            
            if self.lstm_norm != 'none':
                h_in = self.lstm_norm_in(h_in)
                h_out = self.lstm_norm_out(h_out)
            
            edges_emb = torch.cat([h_in, h_out], 1)
            
        else:
            # Original pooling logic (without cross-attention)
            if self.rnn_agg == 'last':
                h_in = h_in.squeeze(0)
                h_out = h_out.squeeze(0)
                if self.lstm_norm != 'none':
                    h_in = self.lstm_norm_in(h_in)
                    h_out = self.lstm_norm_out(h_out)
                edges_emb = torch.cat([h_in, h_out], 1)
            else:
                if self.rnn_agg == 'max':
                    edges_in = torch.max(edges_in_unpacked, dim=1)[0]
                    edges_out = torch.max(edges_out_unpacked, dim=1)[0]
                    edges_emb = torch.cat([edges_in, edges_out], 1)
                elif self.rnn_agg == 'mean':
                    edges_in = torch.mean(edges_in_unpacked, dim=1)
                    edges_out = torch.mean(edges_out_unpacked, dim=1)
                    edges_emb = torch.cat([edges_in, edges_out], 1)
                elif self.rnn_agg == 'sum':
                    edges_in = torch.sum(edges_in_unpacked, dim=1)
                    edges_out = torch.sum(edges_out_unpacked, dim=1)
                    edges_emb = torch.cat([edges_in, edges_out], 1)
        
        x = edges_emb 
        
        # encode
        for i, conv in enumerate(self.encoder):
            if '_e' in self.encoder_layer:
                x, att = conv(x, edge_index, edge_attr)
            elif self.encoder_layer != 'mlp':
                x, att = conv(x, edge_index)
            else:
                x = conv(x)
            if self.gnn_norm != 'none':
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == 0:
                firsta = att.clone().detach().cpu()
        gnn_emb = x.clone()
        
        # decode
        for i, de in enumerate(self.decoder):
            x = de(x)
            if i != len(self.decoder)-1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.out_channels != 1:
            x = F.log_softmax(x, dim=1)
            
        return x, firsta

'''
class Binary_Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rnn_in_channels, encoder_layer='gcn', decoder='mlp', rnn='gru', rnn_agg = 'last',num_layers=2,
                 decoder_layers = 1, dropout=0.5, bias=True, save_mem=True, use_bn=True, concat_feature=1, emb_first=1, heads=1,lstm_norm='ln', gnn_norm = 'bn', graph_op='',aggr='add'):
        super(Binary_Classifier, self).__init__()
        self.rnn_agg = rnn_agg
        self.rnn = rnn
        self.concat_feature = concat_feature
        self.emb_first = emb_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_layers = encoder_layers = num_layers
        self.decoder_layers = decoder_layers
        self.lstm_norm = lstm_norm
        self.gnn_norm = gnn_norm
        self.graph_op = graph_op
        rnn_out_channels = int(hidden_channels/2)
        # Initialize LSTM part
        if emb_first:
            self.lstm_emb_in = nn.Linear(rnn_in_channels, rnn_out_channels)
            self.lstm_emb_out = nn.Linear(rnn_in_channels, rnn_out_channels)
            if self.lstm_norm == 'bn':
                self.lstm_emb_norm_in =  nn.BatchNorm1d(rnn_out_channels)
                self.lstm_emb_norm_out =  nn.BatchNorm1d(rnn_out_channels)
            elif self.lstm_norm == 'ln':
                self.lstm_emb_norm_in =  nn.LayerNorm(rnn_out_channels)
                self.lstm_emb_norm_out =  nn.LayerNorm(rnn_out_channels)
            if rnn == 'lstm':
                self.lstm_in = nn.LSTM(rnn_out_channels, rnn_out_channels)
                self.lstm_out = nn.LSTM(rnn_out_channels, rnn_out_channels)
            elif rnn == 'gru':
                self.lstm_in = nn.GRU(rnn_out_channels, rnn_out_channels)
                self.lstm_out = nn.GRU(rnn_out_channels, rnn_out_channels)
        else:
            if rnn == 'lstm':
                self.lstm_in = nn.LSTM(rnn_in_channels, rnn_out_channels)
                self.lstm_out = nn.LSTM(rnn_in_channels, rnn_out_channels)
            elif rnn == 'gru':
                self.lstm_in = nn.GRU(rnn_in_channels, rnn_out_channels)
                self.lstm_out = nn.GRU(rnn_in_channels, rnn_out_channels)
        # Initialize GNN part
        self.encoder = nn.ModuleList()
        self.encoder_layer = encoder_layer
        use_rnn = 1
        if 'dualcata' in encoder_layer:
            atten_hidden =  encoder_layer.split('-')[-1]
            if atten_hidden.isdigit():
                atten_hidden = int(atten_hidden)
            else:
                atten_hidden = 16
                
            self.encoder.append(
                DualCATAConv(hidden_channels, hidden_channels, bias=bias, atten_hidden=atten_hidden,aggr=aggr))
            for _ in range(encoder_layers-1):
                self.encoder.append(
                    DualCATAConv(hidden_channels, hidden_channels, bias=bias, atten_hidden=atten_hidden,aggr=aggr))
        else:
            raise NameError(f'{encoder_layer} is not implemented!')
        
        # Initialize decoder
        self.decoder = nn.ModuleList()
        for _ in range(decoder_layers-1):
            self.decoder.append(nn.Linear(hidden_channels, hidden_channels))
        self.decoder.append(nn.Linear(hidden_channels, out_channels))
        
        # Initialize other modules
        self.dropout = dropout
        self.activation = F.relu
        # Normalization layer after each encoder layer
        self.bns = nn.ModuleList()
        if self.lstm_norm == 'bn':
            self.lstm_norm_in =  nn.BatchNorm1d(rnn_out_channels)
            self.lstm_norm_out =  nn.BatchNorm1d(rnn_out_channels)
        elif self.lstm_norm == 'ln':
            self.lstm_norm_in =  nn.LayerNorm(rnn_out_channels)
            self.lstm_norm_out =  nn.LayerNorm(rnn_out_channels)
        if self.gnn_norm == 'ln':
            for _ in range(self.encoder_layers):
                self.bns.append(nn.LayerNorm(hidden_channels))
        elif self.gnn_norm == 'bn':
            for _ in range(self.encoder_layers):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                
    def forward(self, in_pack, out_pack, lens_in, lens_out, edge_index = None, edge_attr = None): 
#         t0 = time.time()
        # generate lstm embeddings
        if self.emb_first:
#             in_pack, lens_in = pad_packed_sequence(in_pack)
#             out_pack, lens_out = pad_packed_sequence(out_pack)
            in_pack = self.lstm_emb_in(in_pack)
            in_pack = self.lstm_emb_norm_in(in_pack)
            out_pack = self.lstm_emb_out(out_pack)
            out_pack = self.lstm_emb_norm_out(out_pack)
#             tpc = time.time()
#             print(in_pack.shape)
            in_pack = pack_padded_sequence(in_pack, lens_in.cpu(), batch_first=True, enforce_sorted=False)
            out_pack = pack_padded_sequence(out_pack, lens_out.cpu(), batch_first=True, enforce_sorted=False)
        if self.rnn_agg == 'last':
            if self.rnn == 'lstm':
                edges_in, (h_in,c_in)  = self.lstm_in(in_pack)
                edges_out, (h_out,c_out)  = self.lstm_out(out_pack)
            elif self.rnn == 'gru':
                edges_in, h_in  = self.lstm_in(in_pack)
                edges_out, h_out  = self.lstm_out(out_pack)
            h_in = h_in.squeeze(0)
            h_out = h_out.squeeze(0)
            if self.lstm_norm != 'none':
                h_in = self.lstm_norm_in(h_in)
                h_out = self.lstm_norm_out(h_out)
            edges_emb = torch.cat([h_in, h_out],1 )
        else:
            edges_in, *_  = self.lstm_in(in_pack)
            edges_out, *_  = self.lstm_out(out_pack)
            edges_in = pad_packed_sequence(edges_in)[0]
            edges_out = pad_packed_sequence(edges_out)[0]
            if self.rnn_agg == 'max':
                edges_in = torch.max(edges_in, dim=0)[0]
                edges_out = torch.max(edges_out, dim=0)[0]
                edges_emb = torch.cat([edges_in, edges_out],1 )
            if self.rnn_agg == 'mean':
                edges_in = torch.mean(edges_in, dim=0)
                edges_out = torch.mean(edges_out, dim=0)
                edges_emb = torch.cat([edges_in, edges_out],1 )
            if self.rnn_agg == 'sum':
                edges_in = torch.sum(edges_in, dim=0)
                edges_out = torch.sum(edges_out, dim=0)
                edges_emb = torch.cat([edges_in, edges_out],1 )
        x = edges_emb 
#         if 'D' in self.graph_op: # MultiDi to Directed
#             edge_index = torch.unique(edge_index.t(), dim=0).t()
#         if 'S' in self.graph_op: # Remove self-loops
#             edge_index, _ = remove_self_loops(edge_index)
#         if 'U' in self.graph_op: # Directed to Undirected
#             edge_index = to_undirected(edge_index)
        # encode
        for i, conv in enumerate(self.encoder):
            if '_e' in self.encoder_layer:
                x, att= conv(x, edge_index, edge_attr)
            elif self.encoder_layer != 'mlp':
                x, att = conv(x, edge_index)
            else:
                x = conv(x)
            if self.gnn_norm  != 'none':
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == 0:
                firsta = att.clone().detach().cpu()
        gnn_emb = x.clone()
        # decode
        for i, de in enumerate(self.decoder):
            x = de(x)
            if  i != len(self.decoder)-1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.out_channels != 1:
            x = F.log_softmax(x, dim=1)
        return x, firsta
    
''' 
    
