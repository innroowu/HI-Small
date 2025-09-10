# 在 data_loader.py 中添加以下修改

import pandas as pd
import torch
import numpy as np
import os
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timedelta
import pickle
import json

# 新增導入
try:
    from pattern_analyzer import FraudPatternAnalyzer
    PATTERN_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: pattern_analyzer not found, pattern integration disabled")
    PATTERN_ANALYZER_AVAILABLE = False
    FraudPatternAnalyzer = None

class FraudGTDataLoader:
    """FraudGT資料載入和轉換器 - 修正版本支援基於patterns.txt的標籤生成"""
    
    def __init__(self, data_dir, prediction_window_days=30, use_patterns=True):
        self.data_dir = data_dir
        self.prediction_window_days = prediction_window_days
        self.use_patterns = use_patterns and PATTERN_ANALYZER_AVAILABLE
        
        # 初始化模式分析器
        if self.use_patterns:
            self.pattern_analyzer = FraudPatternAnalyzer(data_dir)
        else:
            self.pattern_analyzer = None
    
    def convert_to_diam_format(self, accounts_file='HI-Small_accounts.csv', 
                              transactions_file='HI-Small_Trans.csv'):
        """完整的轉換流程 - 修正版本"""
        print("Loading CSV data...")
        accounts_df, trans_df = self.load_csv_data(accounts_file, transactions_file)
        
        print("Building graph structure...")
        graph_data = self.build_graph_data(accounts_df, trans_df)
        
        # 修正：優先使用patterns.txt生成標籤，而不是CSV
        if self.use_patterns and self.pattern_analyzer:
            print("Generating labels from patterns.txt...")
            future_risk_labels, temporal_splits = self.generate_labels_from_patterns(
                graph_data['edge_index'], 
                graph_data['edge_timestamps'], 
                graph_data['num_nodes'],
                graph_data['account_mapping']['account_to_id']
            )
            
            # 使用基於patterns的temporal splits
            train_mask, val_mask, test_mask = temporal_splits
        else:
            print("Warning: No patterns available, using CSV labels (likely all zeros)...")
            future_risk_labels = self.generate_future_risk_labels_from_csv(
                graph_data['edge_index'], 
                graph_data['edge_labels'], 
                graph_data['edge_timestamps'], 
                graph_data['num_nodes']
            )
            
            print("Creating temporal splits...")
            train_mask, val_mask, test_mask = self.create_temporal_splits(
                graph_data['edge_index'], 
                graph_data['edge_timestamps'], 
                graph_data['num_nodes']
            )
        
        # 建立最終的Data對象
        data = Data()
        data.edge_index = graph_data['edge_index']
        data.edge_attr = graph_data['edge_attr']
        data.edge_timestamps = graph_data['edge_timestamps']
        data.edge_labels = graph_data['edge_labels']
        data.num_nodes = graph_data['num_nodes']
        
        # 節點標籤和mask
        data.y = future_risk_labels
        data.labels = future_risk_labels
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        # 有標籤的節點mask
        labeled_mask = future_risk_labels >= 0
        data.train_label = data.train_mask & labeled_mask
        data.val_label = data.val_mask & labeled_mask
        data.test_label = data.test_mask & labeled_mask
        
        # 保存元數據
        data.account_mapping = graph_data['account_mapping']
        data.encoders = graph_data['encoders']
        
        # 整合模式信息（新增）
        if self.use_patterns:
            print("Integrating pattern information...")
            try:
                data = self.pattern_analyzer.integrate_patterns_into_data(
                    data, data.account_mapping['account_to_id']
                )
                print("Pattern integration completed successfully!")
            except Exception as e:
                print(f"Pattern integration failed: {e}")
                print("Continuing with basic features...")
        
        print(f"Conversion completed!")
        print(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        print(f"Future risk labels: {future_risk_labels.sum().item()} positive out of {labeled_mask.sum().item()} labeled")
        print(f"Train/Val/Test nodes: {data.train_label.sum().item()}/{data.val_label.sum().item()}/{data.test_label.sum().item()}")
        
        if hasattr(data, 'pattern_stats'):
            print(f"Pattern statistics: {data.pattern_stats}")
        
        return data
    
    def generate_labels_from_patterns(self, edge_index, edge_timestamps, num_nodes, account_to_id):
        """基於patterns.txt生成未來風險標籤和時間分割"""
        print("Generating future risk labels from patterns.txt...")
        
        # 解析patterns檔案
        patterns_data = self.pattern_analyzer.parse_patterns_file()
        
        if not patterns_data:
            print("No patterns found, falling back to CSV method")
            return self.generate_future_risk_labels_from_csv(
                edge_index, torch.zeros(edge_index.size(1)), edge_timestamps, num_nodes
            ), self.create_temporal_splits(edge_index, edge_timestamps, num_nodes)
        
        # 創建ID到帳戶的反向映射
        id_to_account = {v: k for k, v in account_to_id.items()}
        
        # 初始化標籤
        future_risk_labels = torch.zeros(num_nodes, dtype=torch.long)
        pattern_transaction_times = {}  # 記錄每個帳戶的模式交易時間
        
        # 處理每個洗錢模式
        for pattern in patterns_data:
            pattern_transactions = pattern['transactions']
            
            # 將模式中的交易按時間排序
            pattern_tx_with_time = []
            for tx in pattern_transactions:
                try:
                    timestamp = pd.to_datetime(tx['TimeStamp'])
                    pattern_tx_with_time.append((timestamp, tx))
                except:
                    continue
            
            pattern_tx_with_time.sort(key=lambda x: x[0])
            
            # 記錄參與此模式的帳戶
            pattern_accounts = set()
            for timestamp, tx in pattern_tx_with_time:
                from_account = (tx['From Bank'], tx['Account'])
                to_account = (tx['To Bank'], tx['Account.1'])
                
                if from_account in account_to_id:
                    pattern_accounts.add(account_to_id[from_account])
                if to_account in account_to_id:
                    pattern_accounts.add(account_to_id[to_account])
            
            # 設定預測邏輯：從模式開始前的正常行為預測未來風險
            if pattern_tx_with_time:
                pattern_start_time = pattern_tx_with_time[0][0]
                pattern_end_time = pattern_tx_with_time[-1][0]
                
                # 在模式開始前30天內的交易可以被用來預測風險
                prediction_start = pattern_start_time - timedelta(days=self.prediction_window_days)
                
                for account_id in pattern_accounts:
                    # 標記這些帳戶為未來風險
                    future_risk_labels[account_id] = 1
                    
                    # 記錄模式時間用於分割
                    if account_id not in pattern_transaction_times:
                        pattern_transaction_times[account_id] = []
                    pattern_transaction_times[account_id].append({
                        'prediction_start': prediction_start,
                        'pattern_start': pattern_start_time,
                        'pattern_end': pattern_end_time
                    })
        
        # 創建基於模式的時間分割
        train_mask, val_mask, test_mask = self.create_pattern_based_temporal_splits(
            edge_index, edge_timestamps, num_nodes, pattern_transaction_times, account_to_id
        )
        
        print(f"Generated {future_risk_labels.sum().item()} positive risk labels from {len(patterns_data)} patterns")
        
        return future_risk_labels, (train_mask, val_mask, test_mask)
    
    def create_pattern_based_temporal_splits(self, edge_index, edge_timestamps, num_nodes, 
                                       pattern_transaction_times, account_to_id):
        """修正版本：確保每個分割都有正負樣本"""
        print("Creating pattern-based temporal splits...")
        
        # 找到整體時間範圍
        min_timestamp = edge_timestamps.min().item()
        max_timestamp = edge_timestamps.max().item()
        total_time_span = max_timestamp - min_timestamp
        
        # 為每個節點找到其最後一次交易時間
        node_last_time = torch.full((num_nodes,), min_timestamp)
        for node_id in range(num_nodes):
            node_edges = (edge_index[0] == node_id) | (edge_index[1] == node_id)
            if node_edges.any():
                node_last_time[node_id] = edge_timestamps[node_edges].max()
        
        # 定義時間分割點
        train_end_time = min_timestamp + total_time_span * 0.6
        val_end_time = min_timestamp + total_time_span * 0.8
        
        # 基本時間分割（包含所有節點，不論是否有模式）
        train_mask = node_last_time <= train_end_time
        val_mask = (node_last_time > train_end_time) & (node_last_time <= val_end_time)
        test_mask = node_last_time > val_end_time
        
        print(f"Initial splits - Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
        
        # 確保每個分割都有足夠的節點
        min_nodes_per_split = max(100, num_nodes // 50)  # 至少100個或總數的2%
        
        # 如果某個分割節點太少，重新分配
        if val_mask.sum() < min_nodes_per_split:
            print(f"Validation set too small ({val_mask.sum()}), redistributing...")
            # 從訓練集移動一些節點到驗證集
            train_indices = torch.where(train_mask)[0]
            if len(train_indices) > min_nodes_per_split:
                # 隨機選擇一些訓練節點移到驗證集
                move_to_val = train_indices[torch.randperm(len(train_indices))[:min_nodes_per_split]]
                train_mask[move_to_val] = False
                val_mask[move_to_val] = True
        
        if test_mask.sum() < min_nodes_per_split:
            print(f"Test set too small ({test_mask.sum()}), redistributing...")
            # 從訓練集移動一些節點到測試集
            train_indices = torch.where(train_mask)[0]
            if len(train_indices) > min_nodes_per_split:
                move_to_test = train_indices[torch.randperm(len(train_indices))[:min_nodes_per_split]]
                train_mask[move_to_test] = False
                test_mask[move_to_test] = True
        
        print(f"Redistributed splits - Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
        
        return train_mask, val_mask, test_mask
    
    def generate_future_risk_labels_from_csv(self, edge_index, edge_labels, edge_timestamps, num_nodes):
        """原始的CSV方法（備用）"""
        print("Warning: Using CSV method for label generation (likely all zeros)")
        
        future_risk_labels = torch.zeros(num_nodes, dtype=torch.long)
        max_time = edge_timestamps.max().item()
        
        for node_id in range(num_nodes):
            # 找到該節點參與的所有交易
            node_edges_as_sender = (edge_index[0] == node_id)
            node_edges_as_receiver = (edge_index[1] == node_id)
            node_edges = node_edges_as_sender | node_edges_as_receiver
            
            if not node_edges.any():
                continue
                
            node_timestamps = edge_timestamps[node_edges]
            node_edge_labels = edge_labels[node_edges]
            
            # 找到該節點的正常交易時間點
            normal_transactions_mask = node_edge_labels == 0
            if not normal_transactions_mask.any():
                continue
                
            normal_timestamps = node_timestamps[normal_transactions_mask]
            
            # 對每個正常交易時間點，檢查未來窗口內是否有洗錢交易
            for normal_time in normal_timestamps:
                future_window_start = normal_time.item()
                future_window_end = min(future_window_start + self.prediction_window_days, max_time)
                
                future_window_mask = (node_timestamps > future_window_start) & \
                                   (node_timestamps <= future_window_end)
                
                if future_window_mask.any():
                    future_labels = node_edge_labels[future_window_mask]
                    if (future_labels == 1).any():  # 1表示洗錢
                        future_risk_labels[node_id] = 1
                        break
        
        return future_risk_labels
    
    # 其他方法保持不變...
    def detect_column_names(self, df):
        """自動檢測CSV檔案的欄位名稱"""
        columns = df.columns.tolist()
        print(f"Detected columns: {columns}")
        
        # 時間戳欄位的可能名稱
        timestamp_candidates = ['TimeStamp', 'Timestamp', 'timestamp', 'Time', 'time', 'Date', 'date']
        timestamp_col = None
        for candidate in timestamp_candidates:
            if candidate in columns:
                timestamp_col = candidate
                break
        
        if timestamp_col is None:
            # 如果找不到，嘗試包含時間相關關鍵字的欄位
            for col in columns:
                if any(keyword in col.lower() for keyword in ['time', 'date']):
                    timestamp_col = col
                    break
        
        # 洗錢標籤欄位的可能名稱
        laundering_candidates = ['Is Laundering', 'Is_Laundering', 'laundering', 'Laundering', 'Label', 'label']
        laundering_col = None
        for candidate in laundering_candidates:
            if candidate in columns:
                laundering_col = candidate
                break
        
        return timestamp_col, laundering_col, columns
    
    def load_csv_data(self, accounts_file='HI-Small_accounts.csv', transactions_file='HI-Small_Trans.csv'):
        """載入CSV檔案"""
        accounts_path = os.path.join(self.data_dir, accounts_file)
        transactions_path = os.path.join(self.data_dir, transactions_file)
        
        if not os.path.exists(accounts_path):
            raise FileNotFoundError(f"Accounts file not found: {accounts_path}")
        if not os.path.exists(transactions_path):
            raise FileNotFoundError(f"Transactions file not found: {transactions_path}")
            
        # 載入帳戶資料
        accounts_df = pd.read_csv(accounts_path)
        print(f"Loaded {len(accounts_df)} accounts")
        
        # 載入交易資料
        trans_df = pd.read_csv(transactions_path)
        print(f"Loaded {len(trans_df)} transactions")
        
        # 自動檢測欄位名稱
        timestamp_col, laundering_col, all_columns = self.detect_column_names(trans_df)
        
        if timestamp_col is None:
            print("Available columns:", all_columns)
            raise ValueError("Could not find timestamp column. Please check the CSV file structure.")
        
        if laundering_col is None:
            print("Available columns:", all_columns)
            print("Warning: Could not find laundering label column. Will set all to 0.")
            # 如果找不到洗錢標籤欄位，創建一個全為0的欄位
            trans_df['Is Laundering'] = 0
            laundering_col = 'Is Laundering'
            
        print(f"Using timestamp column: {timestamp_col}")
        print(f"Using laundering label column: {laundering_col}")
        
        # 處理時間戳
        trans_df[timestamp_col] = pd.to_datetime(trans_df[timestamp_col])
        trans_df = trans_df.sort_values(timestamp_col).reset_index(drop=True)
        
        # 統一欄位名稱
        trans_df = trans_df.rename(columns={
            timestamp_col: 'TimeStamp',
            laundering_col: 'Is Laundering'
        })
        
        return accounts_df, trans_df
    
    def create_account_mapping(self, trans_df):
        """創建帳戶到節點ID的映射"""
        all_accounts = set()
        
        # 使用統一後的欄位名稱
        for _, row in trans_df.iterrows():
            from_account = (row['From Bank'], row['Account'])
            to_account = (row['To Bank'], row['Account.1'])
            
            all_accounts.add(from_account)
            all_accounts.add(to_account)
        
        all_accounts = list(all_accounts)
        account_to_id = {acc: i for i, acc in enumerate(all_accounts)}
        id_to_account = {i: acc for acc, i in account_to_id.items()}
        
        return account_to_id, id_to_account, trans_df
    
    def process_edge_features(self, trans_df):
        """處理邊特徵"""
        # 檢查是否有Payment Format欄位（注意是Format不是Formats）
        if 'Payment Format' in trans_df.columns:
            trans_df = trans_df.rename(columns={'Payment Format': 'Payment Formats'})
        
        # 處理貨幣編碼
        currency_encoder = LabelEncoder()
        all_currencies = list(trans_df['Receiving Currency'].unique()) + \
                        list(trans_df['Payment Currency'].unique())
        currency_encoder.fit(all_currencies)
        
        # 處理支付格式編碼
        payment_format_encoder = LabelEncoder()
        payment_format_encoder.fit(trans_df['Payment Formats'].unique())
        
        return currency_encoder, payment_format_encoder, trans_df
    
    def build_graph_data(self, accounts_df, trans_df):
        """將表格資料轉換為圖結構"""
        # 創建帳戶映射
        account_to_id, id_to_account, trans_df = self.create_account_mapping(trans_df)
        num_nodes = len(account_to_id)
        
        # 處理編碼器
        currency_encoder, payment_format_encoder, trans_df = self.process_edge_features(trans_df)
        
        # 建立邊和邊屬性
        edge_list = []
        edge_attributes = []
        edge_timestamps = []
        edge_labels = []
        
        min_timestamp = trans_df['TimeStamp'].min()
        
        for idx, row in trans_df.iterrows():
            from_account = (row['From Bank'], row['Account'])
            to_account = (row['To Bank'], row['Account.1'])
            
            from_id = account_to_id[from_account]
            to_id = account_to_id[to_account]
            
            edge_list.append([from_id, to_id])
            
            # 建立邊屬性向量
            edge_attr = [
                row['Amount Received'],
                row['Amount Paid'],
                currency_encoder.transform([row['Receiving Currency']])[0],
                currency_encoder.transform([row['Payment Currency']])[0],
                payment_format_encoder.transform([row['Payment Formats']])[0],
                row['From Bank'],
                row['To Bank']
            ]
            edge_attributes.append(edge_attr)
            
            # 時間戳（轉換為天數）
            timestamp = (row['TimeStamp'] - min_timestamp).total_seconds() / (24 * 3600)
            edge_timestamps.append(timestamp)
            
            # 邊標籤（0: 正常, 1: 洗錢）
            edge_labels.append(row['Is Laundering'])
        
        # 轉換為張量
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
        edge_timestamps = torch.tensor(edge_timestamps, dtype=torch.float)
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        
        # 正規化邊屬性
        scaler = StandardScaler()
        edge_attr_normalized = torch.tensor(scaler.fit_transform(edge_attr), dtype=torch.float)
        
        # 將時間戳添加到邊屬性
        edge_attr_final = torch.cat([edge_attr_normalized, edge_timestamps.unsqueeze(1)], dim=1)
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr_final,
            'edge_timestamps': edge_timestamps,
            'edge_labels': edge_labels,
            'num_nodes': num_nodes,
            'account_mapping': {
                'account_to_id': account_to_id,
                'id_to_account': id_to_account
            },
            'encoders': {
                'currency_encoder': currency_encoder,
                'payment_format_encoder': payment_format_encoder,
                'scaler': scaler
            }
        }
    
    def create_temporal_splits(self, edge_index, edge_timestamps, num_nodes, 
                            train_ratio=0.6, val_ratio=0.2):
        """修正版本的時間分割：確保平衡分佈"""
        print("Creating balanced temporal splits...")
        
        # 計算每個節點的最後交易時間
        node_last_time = torch.zeros(num_nodes)
        for node_id in range(num_nodes):
            node_edges = (edge_index[0] == node_id) | (edge_index[1] == node_id)
            if node_edges.any():
                node_last_time[node_id] = edge_timestamps[node_edges].max()
        
        # 使用分位數而不是絕對時間來分割，確保更平衡的分佈
        active_nodes = node_last_time > 0
        active_times = node_last_time[active_nodes]
        
        if len(active_times) == 0:
            # 如果沒有活躍節點，隨機分割
            indices = torch.randperm(num_nodes)
            train_size = int(num_nodes * train_ratio)
            val_size = int(num_nodes * val_ratio)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)  
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size+val_size]] = True
            test_mask[indices[train_size+val_size:]] = True
        else:
            # 使用時間分位數分割
            train_quantile = torch.quantile(active_times, train_ratio)
            val_quantile = torch.quantile(active_times, train_ratio + val_ratio)
            
            train_mask = (node_last_time > 0) & (node_last_time <= train_quantile)
            val_mask = (node_last_time > train_quantile) & (node_last_time <= val_quantile)
            test_mask = (node_last_time > val_quantile)
            
            # 處理沒有交易的節點 - 隨機分配
            inactive_nodes = node_last_time == 0
            if inactive_nodes.any():
                inactive_indices = torch.where(inactive_nodes)[0]
                perm = torch.randperm(len(inactive_indices))
                
                inactive_train_size = int(len(inactive_indices) * train_ratio)
                inactive_val_size = int(len(inactive_indices) * val_ratio)
                
                train_mask[inactive_indices[perm[:inactive_train_size]]] = True
                val_mask[inactive_indices[perm[inactive_train_size:inactive_train_size+inactive_val_size]]] = True
                test_mask[inactive_indices[perm[inactive_train_size+inactive_val_size:]]] = True
        
        print(f"Balanced splits - Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
        return train_mask, val_mask, test_mask
    
    def save_data(self, data, filename):
        """保存處理後的數據"""
        filepath = os.path.join(self.data_dir, filename)
        torch.save(data, filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename):
        """載入處理後的數據"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = torch.load(filepath)
        print(f"Data loaded from {filepath}")
        return data