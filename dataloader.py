import pickle 
import json
import os
    
    
from torch_geometric.utils import subgraph
from torch_geometric.transforms import RandomNodeSplit
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from typing import Union, List, Dict, Tuple, Callable, Optional
from torch_geometric.typing import NodeType, EdgeType
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from tqdm import tqdm
from torch_geometric.data import Data, HeteroData
from torch_geometric.sampler.utils import to_csc, to_hetero_csc
from torch_geometric.loader.utils import filter_data
import torch_geometric
import numpy as np
import pandas as pd

import random

dir='/share/nas169/innroowu/DIAM_AML1/data/HI_Small'
data_dir='/share/nas169/innroowu/DIAM_AML1/data/HI_Small'

def load_pickle(fname):
    with open(os.path.join(data_dir,fname), 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    with open( os.path.join(data_dir,name)+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def save_json(data, name):
    with open( os.path.join(data_dir,name)+'.json', 'w') as f:
        json.dump(data,f)
def load_json(fname):
    with open( os.path.join(data_dir,fname), 'r') as f:
        return json.load(f)

def subdata(data: torch_geometric.data.data.Data, subset, subedges=None, relabel_nodes=True):
    device = data.edge_index.device
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        node_mask = subset
        num_nodes = node_mask.size(0)

        if relabel_nodes:
            node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                                   device=device)
            node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[subset] = 1

        if relabel_nodes:
            node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_idx[subset] = torch.arange(subset.size(0), device=device)
            
    sub_data = Data()
    
    if subedges is not None:
        if subedges.dtype == torch.bool:
            assert subedges.shape[0] == num_edges
        else:
            assert subedges.max() < num_edges
            
    # Get subgraph nodes and edges feature
    for key, item in data:
        if key in ['num_nodes', 'edge_index']:
            continue
        if isinstance(item, Tensor) and item.size(0) == num_nodes:
            sub_data[key] = item[subset]
        elif isinstance(item, Tensor) and item.size(0) == num_edges:
            if subedges is None:
                edge_index, sub_data[key] = subgraph(subset, data.edge_index, data[key], relabel_nodes=relabel_nodes)  
            else:
                sub_data[key] = item[subedges]
        else:
            sub_data[key] = item
    if subedges is None:
        sub_data.edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=relabel_nodes)   
    else:
        edge_index = data.edge_index[:, subedges]
        if relabel_nodes:
            edge_index = node_idx[edge_index]
        sub_data.edge_index = edge_index
    return sub_data


def temporal_split_transductive(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    嚴謹的 Transductive 時間切割：
    - 所有節點在所有階段都可見
    - 但每個階段只能使用到該時間點的邊
    - 節點標籤的訓練/驗證/測試根據其最後出現時間決定
    
    訓練階段: 可見所有節點 + 使用 t < t1 的邊 + 訓練 t < t1 的節點標籤
    驗證階段: 可見所有節點 + 使用 t < t2 的邊 + 評估 t1 ≤ t < t2 的節點
    測試階段: 可見所有節點 + 使用所有邊 + 評估 t ≥ t2 的節點
    
    Args:
        data: PyTorch Geometric Data 物件
        train_ratio: 訓練集時間比例
        val_ratio: 驗證集時間比例
        test_ratio: 測試集時間比例
    
    Returns:
        train_mask, val_mask, test_mask (節點層級的 mask)
        train_edge_mask, val_edge_mask, test_edge_mask (邊層級的 mask)
    """
    # 從 edge_attr 取得時間戳（最後一個維度）
    #timestamps = data.edge_attr[:, -1]
    timestamps = data.edge_attr[:, 4]
    
    # 取得交易的排序索引（按時間）
    sorted_edge_indices = torch.argsort(timestamps)
    
    # 計算切分點（基於交易數量）
    num_edges = len(sorted_edge_indices)
    train_edge_end = int(num_edges * train_ratio)
    val_edge_end = int(num_edges * (train_ratio + val_ratio))
    
    print(f"\n{'='*70}")
    print(f"嚴謹 Transductive 時間切割")
    print(f"{'='*70}")
    
    # 計算時間閾值
    train_time_threshold = timestamps[sorted_edge_indices[train_edge_end - 1]].item()
    val_time_threshold = timestamps[sorted_edge_indices[val_edge_end - 1]].item()
    
    print(f"時間閾值:")
    print(f"  t1 (train/val boundary): {train_time_threshold:.2f}")
    print(f"  t2 (val/test boundary):  {val_time_threshold:.2f}")
    
    # 建立邊的 mask（基於時間閾值）
    train_edge_mask = timestamps < train_time_threshold
    val_edge_mask = timestamps < val_time_threshold  # 驗證時可用 t < t2 的所有邊
    test_edge_mask = torch.ones_like(timestamps, dtype=torch.bool)  # 測試時可用所有邊
    
    print(f"\n邊的可用性:")
    print(f"  訓練階段可用邊: {train_edge_mask.sum().item()} ({train_edge_mask.sum().item()/num_edges*100:.1f}%)")
    print(f"  驗證階段可用邊: {val_edge_mask.sum().item()} ({val_edge_mask.sum().item()/num_edges*100:.1f}%)")
    print(f"  測試階段可用邊: {test_edge_mask.sum().item()} ({test_edge_mask.sum().item()/num_edges*100:.1f}%)")
    
    # 為每個節點找出它最後出現的時間
    num_nodes = data.y.shape[0]
    node_last_time = torch.full((num_nodes,), float('-inf'))
    
    # 遍歷所有邊，記錄每個節點最後出現的時間
    for edge_idx in range(num_edges):
        edge_time = timestamps[edge_idx].item()
        src_node = data.edge_index[0, edge_idx].item()
        dst_node = data.edge_index[1, edge_idx].item()
        node_last_time[src_node] = max(node_last_time[src_node], edge_time)
        node_last_time[dst_node] = max(node_last_time[dst_node], edge_time)
    
    # 根據節點最後出現時間分配到不同集合
    # 訓練: t < t1
    # 驗證: t1 ≤ t < t2
    # 測試: t ≥ t2
    train_mask = node_last_time < train_time_threshold
    val_mask = (node_last_time >= train_time_threshold) & (node_last_time < val_time_threshold)
    test_mask = node_last_time >= val_time_threshold
    
    print(f"\n節點分配（基於最後出現時間）:")
    print(f"  訓練節點: {train_mask.sum().item()} ({train_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"  驗證節點: {val_mask.sum().item()} ({val_mask.sum().item()/num_nodes*100:.1f}%)")
    print(f"  測試節點: {test_mask.sum().item()} ({test_mask.sum().item()/num_nodes*100:.1f}%)")
    
    # 儲存邊的索引和 mask，供後續使用
    data.train_edge_mask = train_edge_mask
    data.val_edge_mask = val_edge_mask
    data.test_edge_mask = test_edge_mask
    
    data.train_time_threshold = train_time_threshold
    data.val_time_threshold = val_time_threshold
    
    print(f"\n重要提示:")
    print(f"  ✓ 所有節點在所有階段都可見")
    print(f"  ✓ 訓練時只能使用 t < t1 的邊進行消息傳遞")
    print(f"  ✓ 驗證時可以使用 t < t2 的邊進行消息傳遞")
    print(f"  ✓ 測試時可以使用所有邊進行消息傳遞")
    print(f"{'='*70}\n")
    
    return train_mask, val_mask, test_mask


def load_data(data_path='./data/HI_Small', use_unlabeled='SEMI', scale='minmax', 
              graph_type='MultiDi', feature_type='edge', train_rate=0.6, 
              anomaly_rate=None, random_state=5211, temporal_split_mode=True):
    """
    載入資料並進行切分
    
    Args:
        temporal_split_mode: True 使用時間切分（預設為嚴謹的 Transductive）
    """
    # fix random seeds
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
    
    data = torch.load(data_path)
    
    if anomaly_rate:
        n_neg = (data.y == 0).sum().item()
        pos_ids = (data.y == 1).nonzero().view(-1).numpy()
        np.random.shuffle(pos_ids)
        drop_pos_ids = pos_ids[int(n_neg*anomaly_rate/(1-anomaly_rate)):]
        data.y[drop_pos_ids] = -1
    
    labels = data.y
    n_nodes = len(labels)
    all_id = np.arange(n_nodes)
    
    # label_mask is used in semi-supervised setting
    if use_unlabeled == 'ALL':
        labels = np.where(labels == -1, 0, labels)
        label_mask = torch.ones(len(labels)).bool()
    elif use_unlabeled == 'NONE': 
        labels += 1 
        nodes_id = labels.nonzero().reshape(-1)
        labels = labels[nodes_id]
        labels -= 1
        data = subdata(data, nodes_id, relabel_nodes=True)
        label_mask = torch.ones(len(labels)).bool()
    elif use_unlabeled == 'SEMI': 
        labels += 1 
        label_id = labels.nonzero().reshape(-1)
        label_mask = torch.zeros(n_nodes)
        label_mask[label_id] = 1
        label_mask = label_mask.bool()
        labels -= 1
        
    n_nodes = len(labels)
    all_id = np.arange(n_nodes)
    
    data['labels'] = torch.tensor(labels, dtype=int)
    
    # Split data: 使用嚴謹的 Transductive Split
    if temporal_split_mode:
        val_ratio = (1 - train_rate) / 2
        test_ratio = (1 - train_rate) / 2
        train_mask, val_mask, test_mask = temporal_split_transductive(
            data, 
            train_ratio=train_rate, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio
        )
    else:
        # Random split (保留原有功能)
        print("=" * 60)
        print("使用 Random Split（隨機切分）")
        print("=" * 60)
        np.random.shuffle(all_id)
        train_id = all_id[:int(n_nodes*train_rate)]
        val_id = all_id[int(n_nodes*0.5): int(n_nodes*(1+0.5)/2)]
        test_id = all_id[int(n_nodes*(1+0.5)/2): -1]
        
        train_mask = torch.zeros(n_nodes)
        train_mask[train_id] = 1
        train_mask = train_mask.bool()
        
        val_mask = torch.zeros(n_nodes)
        val_mask[val_id] = 1
        val_mask = val_mask.bool()
        
        test_mask = torch.zeros(n_nodes)
        test_mask[test_id] = 1
        test_mask = test_mask.bool()
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    
    # Mask nodes which are labeled in SEMI
    data['train_label'] = data.train_mask & label_mask
    data['val_label'] = data.val_mask & label_mask
    data['test_label'] = data.test_mask & label_mask
    
    # 顯示標籤分布
    print(f"\n{'=' * 60}")
    print("標籤分布統計:")
    print(f"{'=' * 60}")
    print(f"訓練集 - 警示: {(data.labels[data.train_label] == 1).sum().item():6d}, "
          f"正常: {(data.labels[data.train_label] == 0).sum().item():6d}, "
          f"未知: {(data.labels[data.train_mask] == -1).sum().item():6d}")
    print(f"驗證集 - 警示: {(data.labels[data.val_label] == 1).sum().item():6d}, "
          f"正常: {(data.labels[data.val_label] == 0).sum().item():6d}, "
          f"未知: {(data.labels[data.val_mask] == -1).sum().item():6d}")
    print(f"測試集 - 警示: {(data.labels[data.test_label] == 1).sum().item():6d}, "
          f"正常: {(data.labels[data.test_label] == 0).sum().item():6d}, "
          f"未知: {(data.labels[data.test_mask] == -1).sum().item():6d}")
    print(f"{'=' * 60}\n")
    
    return data, 2