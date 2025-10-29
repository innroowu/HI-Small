#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
資料處理腳本：將 AML 交易資料轉換為圖結構並儲存為 .pt 檔案
則 node_gfp_features 的最終維度是:
    - 前半部分：該帳戶作為來源的所有交易的 edge_gfp_features平均值。
    - 後半部分：該帳戶作為目的的所有交易的 edge_gfp_features平均值。

"""

import os
import json
import torch
import pandas as pd
import re
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch_geometric.data import Data
from snapml import GraphFeaturePreprocessor


def normalize_account_id(bank_code, account):
    """
    標準化帳戶 ID，移除前導零
    
    Args:
        bank_code: 銀行代碼
        account: 帳號
    
    Returns:
        str: 標準化後的帳戶 ID
    """
    # 將銀行代碼和帳號轉為字串並移除前導零
    bank_code_str = str(bank_code).lstrip('0') or '0'
    account_str = str(account).lstrip('0') or '0'
    return f"{bank_code_str}_{account_str}"


def parse_patterns_file(patterns_path):
    """
    解析 Patterns.txt 檔案，提取洗錢交易資訊
    
    Args:
        patterns_path: Patterns.txt 檔案路徑
    
    Returns:
        list: 洗錢交易列表，每筆交易為 dict
        set: 洗錢相關帳戶集合（已標準化，移除前導零）
    """
    laundering_transactions = []
    laundering_accounts = set()
    
    with open(patterns_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找出所有洗錢區塊
    pattern_blocks = re.findall(
        r'BEGIN LAUNDERING ATTEMPT.*?END LAUNDERING ATTEMPT[^\n]*',
        content,
        re.DOTALL
    )
    
    for block in pattern_blocks:
        lines = block.strip().split('\n')
        for line in lines[1:-1]:  # 跳過 BEGIN 和 END 行
            if line.strip() and not line.startswith('BEGIN') and not line.startswith('END'):
                parts = line.strip().split(',')
                if len(parts) >= 10:
                    from_bank = parts[1].strip()
                    from_account = parts[2].strip()
                    to_bank = parts[3].strip()
                    to_account = parts[4].strip()
                    
                    transaction = {
                        'timestamp': parts[0].strip(),
                        'from_bank': from_bank,
                        'from_account': from_account,
                        'to_bank': to_bank,
                        'to_account': to_account,
                        'amount_received': float(parts[5]),
                        'receiving_currency': parts[6].strip(),
                        'amount_paid': float(parts[7]),
                        'payment_currency': parts[8].strip(),
                        'payment_format': parts[9].strip(),
                        'is_laundering': int(parts[10].strip())
                    }
                    laundering_transactions.append(transaction)
                    
                    # 標準化帳戶 ID（移除前導零）
                    from_acct = normalize_account_id(from_bank, from_account)
                    to_acct = normalize_account_id(to_bank, to_account)
                    laundering_accounts.add(from_acct)
                    laundering_accounts.add(to_acct)
    
    return laundering_transactions, laundering_accounts


def transaction_hop(trans_data, acct_list, hop=1, without_acct_list=None):
    """
    取得警示帳戶的 n-hop 名單
    
    Args:
        trans_data: 交易資料 DataFrame
        acct_list: 帳戶列表
        hop: 跳數
        without_acct_list: 要排除的帳戶列表
    
    Returns:
        set: n-hop 帳戶集合
    """
    acct_hop_dict = set(acct_list)

    for _ in range(hop):
        related_transaction = trans_data[
            (trans_data['from_acct'].isin(acct_hop_dict)) | 
            (trans_data['to_acct'].isin(acct_hop_dict))
        ]
        
        for row in related_transaction.itertuples():
            acct_hop_dict.add(row.from_acct)
            acct_hop_dict.add(row.to_acct)
    
    if without_acct_list is not None:
        acct_hop_dict = acct_hop_dict - set(without_acct_list)

    return acct_hop_dict


def datetime_to_timestamp(timestamp_str):
    """
    將時間戳字串轉換為數值（Unix timestamp 或序數）
    
    Args:
        timestamp_str: 時間戳字串，格式 'YYYY/MM/DD HH:MM'
    
    Returns:
        float: 時間戳數值
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y/%m/%d %H:%M')
        # 轉換為 Unix timestamp（秒）
        return dt.timestamp()
    except:
        return 0.0

def extract_gfp_features(trans_data, acct_id, time_window_days=1):
    """
    使用 Graph Feature Preprocessor 提取圖特徵
    
    Args:
        trans_data: 處理後的交易資料 DataFrame
        acct_id: 帳戶 ID 映射字典
        time_window_days: 時間窗口（天）
    
    Returns:
        node_features: 節點層級特徵 (num_nodes, num_features)
        edge_features: 交易層級特徵 (num_edges, num_features)
    """
    print(f"\n{'='*70}")
    print("使用 Graph Feature Preprocessor 提取圖特徵")
    print(f"{'='*70}")
    
    # 準備 GFP 輸入格式：[transaction_id, source_account_id, target_account_id, timestamp, amount]
    # GFP 需要的是數值型的 account ID
    gfp_input = np.column_stack([
        np.arange(len(trans_data)),           # transaction_id
        trans_data['from_acct_id'].values,    # source_account_id
        trans_data['to_acct_id'].values,      # target_account_id  
        trans_data['txn_timestamp'].values,   # timestamp
        trans_data['txn_amt'].values          # amount (用於統計)
    ])
    
    print(f"GFP 輸入形狀: {gfp_input.shape}")
    print(f"交易數量: {len(gfp_input)}")
    print(f"帳戶數量: {len(acct_id)}")
    
    # 計算時間窗口（秒）
    time_window_seconds = time_window_days * 24 * 3600
    
    # 配置 GFP 參數
    params = {
        "num_threads": 4,
        "time_window": time_window_seconds,
        
        # 節點統計特徵
        "vertex_stats": True,
        "vertex_stats_cols": [4],  # 使用 amount 欄位計算統計
        "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],  # fan, deg, ratio, avg, sum, var, skew, kurtosis
        
        # Fan-in/out patterns
        "fan": True,
        "fan_tw": time_window_seconds,
        "fan_bins": [2, 5, 10],
        
        # Degree patterns  
        "degree": True,
        "degree_tw": time_window_seconds,
        "degree_bins": [2, 5, 10],
        
        # Scatter-gather patterns
        "scatter-gather": True,
        "scatter-gather_tw": time_window_seconds,
        "scatter-gather_bins": [2, 5],
        
        # Temporal cycle patterns
        "temp-cycle": True,
        "temp-cycle_tw": time_window_seconds,
        "temp-cycle_bins": [2, 5],
        
        # Length-constrained simple cycle (optional, 可能較慢)
        "lc-cycle": False,
    }
    
    print("\nGFP 參數配置:")
    print(f"  時間窗口: {time_window_days} 天 ({time_window_seconds} 秒)")
    print(f"  節點統計: 使用 amount 欄位")
    print(f"  模式檢測: fan, degree, scatter-gather, temporal-cycle")
    
    # 初始化並執行 GFP
    print("\n開始提取圖特徵...")
    gfp = GraphFeaturePreprocessor()
    gfp.set_params(params)
    
    # fit_transform 會同時建立圖並提取特徵
    enriched_features = gfp.fit_transform(gfp_input.astype("float64"))
    
    print(f"✓ 特徵提取完成！")
    print(f"  原始特徵數: {gfp_input.shape[1]}")
    print(f"  總特徵數: {enriched_features.shape[1]}")
    print(f"  新增 GFP 特徵數: {enriched_features.shape[1] - gfp_input.shape[1]}")
    
    # 分離原始特徵和 GFP 特徵
    gfp_features_only = enriched_features[:, gfp_input.shape[1]:]  # 只要新增的特徵
    
    # 交易層級特徵 = 每條 edge 的 GFP 特徵
    edge_features = gfp_features_only
    
    # 節點層級特徵 = 聚合該節點所有相關交易的 GFP 特徵
    # 為每個節點計算其作為 source 和 target 的平均特徵
    num_nodes = len(acct_id)
    num_gfp_features = gfp_features_only.shape[1]
    node_features = np.zeros((num_nodes, num_gfp_features * 2))  # *2 for in/out
    """
    print("\n聚合節點層級特徵...")
    for node_id in range(num_nodes):
        # 作為 source 的交易
        src_mask = trans_data['from_acct_id'] == node_id
        if src_mask.any():
            node_features[node_id, :num_gfp_features] = gfp_features_only[src_mask].mean(axis=0)
        
        # 作為 target 的交易
        tgt_mask = trans_data['to_acct_id'] == node_id
        if tgt_mask.any():
            node_features[node_id, num_gfp_features:] = gfp_features_only[tgt_mask].mean(axis=0)
    
    print(f"✓ 節點特徵聚合完成")
    print(f"  節點特徵維度: {node_features.shape}")
    """
    print(f"  交易特徵維度: {edge_features.shape}")
    print(f"{'='*70}\n")
    
    return node_features, edge_features

def main():
    # 設定路徑
    base_path = "./data/HI_Small"  # 請修改為你的資料路徑
    output_path = "./data/HI_Small/results"  # 請修改為輸出路徑
    
    trans_path = os.path.join(base_path, "HI-Small_Trans.csv")
    patterns_path = os.path.join(base_path, "HI-Small_Patterns.txt")
    
    # 建立輸出目錄
    os.makedirs(output_path, exist_ok=True)
    
    print("載入資料...")
    # 載入交易資料
    trans_data = pd.read_csv(trans_path)
    print(f"交易資料: {len(trans_data)} 筆")
    
    # 解析 Patterns 檔案
    print("解析洗錢模式...")
    laundering_transactions, laundering_accounts = parse_patterns_file(patterns_path)
    print(f"洗錢交易: {len(laundering_transactions)} 筆")
    print(f"洗錢相關帳戶: {len(laundering_accounts)} 個")
    
    # 1. 處理帳戶資訊：組合銀行代碼和帳號（移除前導零）
    print("\n處理帳戶資訊...")
    trans_data['from_acct'] = trans_data.apply(
        lambda row: normalize_account_id(row['From Bank'], row['Account']), 
        axis=1
    )
    trans_data['to_acct'] = trans_data.apply(
        lambda row: normalize_account_id(row['To Bank'], row['Account.1']), 
        axis=1
    )
    
    # 2. 轉換帳戶代號為 ID 編號
    print("處理帳戶 ID 映射...")
    acct_list = pd.concat([trans_data['from_acct'], trans_data['to_acct']]).unique()
    
    acct_id = {}
    for i in acct_list:
        acct_id[i] = len(acct_id)
    
    trans_data['from_acct_id'] = trans_data['from_acct'].map(lambda x: acct_id[x])
    trans_data['to_acct_id'] = trans_data['to_acct'].map(lambda x: acct_id[x])
    
    # 3. 處理交易時間（轉換為數值）
    print("處理交易時間...")
    trans_data['txn_timestamp'] = trans_data['Timestamp'].apply(datetime_to_timestamp)
    
    # 4. 處理交易金額（使用 Amount Paid，資料集已經是美元）
    print("處理交易金額...")
    trans_data['txn_amt'] = trans_data['Amount Paid']
    
    # 5. 處理貨幣類型（currency_type 使用 Payment Currency）
    print("處理貨幣類型...")
    currency_id = {}
    for i in trans_data['Payment Currency'].unique():
        currency_id[i] = len(currency_id)
    
    trans_data['currency_id'] = trans_data['Payment Currency'].map(
        lambda x: currency_id[x]
    )
    
    # 6. 處理交易通路（channel_type 使用 Payment Format）
    print("處理交易通路...")
    channel_id = {}
    for i in trans_data['Payment Format'].unique():
        channel_id[i] = len(channel_id)
    
    trans_data['channel_id'] = trans_data['Payment Format'].map(
        lambda x: channel_id[x]
    )
    
    # 7. 判斷是否為自我交易（根據銀行代碼是否相同）
    print("處理自我交易標記...")
    trans_data['is_self_txn_id'] = (
        trans_data['From Bank'].astype(str).str.lstrip('0') == 
        trans_data['To Bank'].astype(str).str.lstrip('0')
    ).astype(int)
    
    # 建構圖結構
    print("\n建構圖結構...")
    
    # Edge index
    from_acct_id = trans_data['from_acct_id'].values
    to_acct_id = trans_data['to_acct_id'].values
    edge_index = torch.tensor([from_acct_id, to_acct_id], dtype=torch.long)
    
    # Edge attributes: [is_self_txn, txn_amt, currency_id, channel_id, timestamp]
    is_self = trans_data['is_self_txn_id'].values
    amt = trans_data['txn_amt'].values
    currency = trans_data['currency_id'].values
    channel = trans_data['channel_id'].values
    timestamp = trans_data['txn_timestamp'].values
    
    edge_attr = torch.tensor(
        [is_self, amt, currency, channel, timestamp], 
        dtype=torch.float
    ).t()
    
    # ===== 使用 GFP 提取圖特徵 =====
    node_gfp_features, edge_gfp_features = extract_gfp_features(
        trans_data, acct_id, time_window_days=1
    )
    
    # 轉換為 torch tensor
    edge_gfp_features = torch.tensor(edge_gfp_features, dtype=torch.float)
    
    # 🔧 新增：整合 GFP 特徵到 edge_attr
    edge_attr = torch.cat([edge_attr, edge_gfp_features], dim=1)
    
    print(f"\n{'='*70}")
    print("Edge Attributes 整合完成")
    print(f"{'='*70}")
    print(f"  原始特徵數: 5 (is_self_txn, txn_amt, currency_id, channel_id, timestamp)")
    print(f"  GFP 特徵數: {edge_gfp_features.shape[1]}")
    print(f"  整合後總特徵數: {edge_attr.shape[1]}")
    print(f"  Edge_attr 形狀: {edge_attr.shape}")
    print(f"{'='*70}\n")
    
    # Labels
    print("處理標籤...")
    
    # 從 laundering_accounts 中提取在 acct_id 中的帳戶
    alert_accts = [acct for acct in laundering_accounts if acct in acct_id]
    
    # 取得警示帳戶的 1-hop 鄰居（排除警示帳戶本身）
    print("計算 1-hop 鄰居...")
    acct_nhop_with_alert = transaction_hop(
        trans_data,
        alert_accts, 
        hop=1, 
        without_acct_list=alert_accts
    )
    
    labels = [-1 for i in range(len(acct_id))]
    
    # 設定警示帳戶標籤為 1
    for acct in alert_accts:
        if acct in acct_id:
            labels[acct_id[acct]] = 1
    
    # 設定 1-hop 鄰居標籤為 0（正常帳戶）
    for acct in acct_nhop_with_alert:
        if acct in acct_id:
            labels[acct_id[acct]] = 0
    
    y = torch.tensor(labels, dtype=torch.long)
    
    print(f"標籤統計 - 警示(1): {labels.count(1)}, 正常(0): {labels.count(0)}, 未知(-1): {labels.count(-1)}")
    

    # 建立 Data 物件
    data = Data(
        edge_index=edge_index, 
        edge_attr=edge_attr,  # 已包含 GFP 特徵
        y=y
    )
    # 注意：node_gfp_features 不再需要，因為我們使用 edge-level GFP
    
    print(f"\n圖結構: {data}")
    print(f"節點數量: {len(acct_id)}")
    print(f"邊數量: {len(edge_index[0])}")
    print(f"邊特徵維度: {edge_attr.shape}")
    
    # 儲存結果
    print("\n儲存結果...")
    
    acct_id_path = os.path.join(output_path, 'acct_id.json')
    data_path = os.path.join(base_path, 'data.pt')
    trans_output_path = os.path.join(output_path, 'processed_transactions.csv')
    laundering_accts_path = os.path.join(output_path, 'laundering_accounts.json')
    metadata_path = os.path.join(output_path, 'metadata.json')
    
    # 儲存帳戶 ID 映射
    with open(acct_id_path, 'w') as f:
        json.dump(acct_id, f, indent=4)
    
    # 儲存洗錢帳戶清單
    with open(laundering_accts_path, 'w') as f:
        json.dump(list(laundering_accounts), f, indent=4)
    
    # 儲存元數據（映射表）
    metadata = {
        'currency_id': currency_id,
        'channel_id': channel_id,
        'num_nodes': len(acct_id),
        'num_edges': len(edge_index[0]),
        'num_laundering_accounts': labels.count(1),
        'edge_attr_description': [
            'is_self_txn',       # dim 0
            'txn_amt',           # dim 1
            'currency_id',       # dim 2
            'channel_id',        # dim 3
            'timestamp',         # dim 4
            'gfp_features'       # dim 5 onwards (包含所有 GFP 特徵)
        ],
        'edge_attr_total_dim': edge_attr.shape[1],
        'edge_attr_original_dim': 5,
        'edge_attr_gfp_dim': edge_gfp_features.shape[1],
        'gfp_time_window_days': 1
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # 儲存 PyTorch Geometric 圖資料
    torch.save(data, data_path)
    
    # 儲存處理後的交易資料
    trans_data.to_csv(trans_output_path, index=False)
    
    print(f"✓ 帳戶 ID 映射已儲存至: {acct_id_path}")
    print(f"✓ 洗錢帳戶清單已儲存至: {laundering_accts_path}")
    print(f"✓ 元數據已儲存至: {metadata_path}")
    print(f"✓ 圖資料已儲存至: {data_path}")
    print(f"✓ 處理後的交易資料已儲存至: {trans_output_path}")
    print("\n完成!")


if __name__ == "__main__":
    main()