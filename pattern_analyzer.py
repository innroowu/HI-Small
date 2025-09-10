import pandas as pd
import torch
import numpy as np
import os
from collections import defaultdict
import re
from datetime import datetime, timedelta

class FraudPatternAnalyzer:
    """最終修正版的模式分析器，解決Bank ID格式不匹配問題"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pattern_types = {
            'FAN-OUT': 0,
            'FAN-IN': 1, 
            'CYCLE': 2,
            'GATHER-SCATTER': 3,
            'SCATTER-GATHER': 4,
            'STACK': 5,
            'BIPARTITE': 6,
            'RANDOM': 7
        }
        
        # 緩存解析結果
        self._cached_patterns = None
        self._cached_accounts = None
        
    def normalize_bank_id(self, bank_id):
        """標準化Bank ID格式：統一轉為整數進行比較"""
        if isinstance(bank_id, str):
            # 移除前導零並轉為整數
            try:
                return int(bank_id.lstrip('0') or '0')
            except ValueError:
                return bank_id
        elif isinstance(bank_id, (int, float)):
            return int(bank_id)
        else:
            return bank_id
    
    def create_normalized_account_key(self, bank_id, account_number):
        """創建標準化的帳戶鍵"""
        normalized_bank = self.normalize_bank_id(bank_id)
        return (normalized_bank, str(account_number))
    
    def parse_patterns_file(self, patterns_file='HI-Small_Patterns.txt'):
        """解析模式檔案"""
        if self._cached_patterns is not None:
            return self._cached_patterns
            
        patterns_path = os.path.join(self.data_dir, patterns_file)
        
        if not os.path.exists(patterns_path):
            print(f"Warning: {patterns_file} not found, pattern integration disabled")
            return []
            
        patterns_data = []
        current_pattern = None
        current_transactions = []
        
        print(f"Parsing pattern file: {patterns_path}")
        
        try:
            with open(patterns_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        continue
                        
                    if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                        pattern_type = self._extract_pattern_type(line)
                        current_pattern = {
                            'pattern_type': pattern_type,
                            'pattern_name': self._get_pattern_name(pattern_type),
                            'line_number': line_num
                        }
                        current_transactions = []
                        
                    elif line.startswith('END LAUNDERING ATTEMPT'):
                        if current_pattern and current_transactions:
                            current_pattern['transactions'] = current_transactions
                            current_pattern['transaction_count'] = len(current_transactions)
                            patterns_data.append(current_pattern)
                        current_pattern = None
                        current_transactions = []
                        
                    elif line and not line.startswith('BEGIN') and not line.startswith('END'):
                        transaction = self._parse_transaction_line(line, line_num)
                        if transaction:
                            current_transactions.append(transaction)
                        
        except Exception as e:
            print(f"Error parsing patterns file: {e}")
            return []
        
        self._cached_patterns = patterns_data
        print(f"Successfully parsed {len(patterns_data)} laundering patterns")
        return patterns_data
    
    def _extract_pattern_type(self, line):
        """從標題行提取模式類型"""
        line_upper = line.upper()
        for pattern_name, pattern_id in self.pattern_types.items():
            if pattern_name in line_upper:
                return pattern_id
        return 7  # 默認為 RANDOM
    
    def _get_pattern_name(self, pattern_type):
        """根據類型ID獲取模式名稱"""
        for name, type_id in self.pattern_types.items():
            if type_id == pattern_type:
                return name
        return 'UNKNOWN'
    
    def _parse_transaction_line(self, line, line_num=0):
        """解析交易行"""
        try:
            line = line.strip().strip('"')
            parts = [part.strip().strip('"') for part in line.split(',')]
            
            if len(parts) < 10:
                return None
                
            try:
                amount_received = float(parts[5]) if parts[5] else 0.0
                amount_paid = float(parts[7]) if parts[7] else 0.0
            except ValueError:
                amount_received = amount_paid = 0.0
            
            try:
                timestamp = pd.to_datetime(parts[0])
            except:
                timestamp = pd.Timestamp.now()
            
            is_laundering = 1  # patterns.txt中的交易都是洗錢交易
            if len(parts) > 10:
                try:
                    is_laundering = int(parts[10])
                except ValueError:
                    is_laundering = 1
            
            return {
                'TimeStamp': timestamp,
                'From Bank': parts[1],
                'Account': parts[2],
                'To Bank': parts[3],
                'Account.1': parts[4],
                'Amount Received': amount_received,
                'Receiving Currency': parts[6],
                'Amount Paid': amount_paid,
                'Payment Currency': parts[8],
                'Payment Formats': parts[9],
                'Is Laundering': is_laundering
            }
        except Exception as e:
            return None
    
    def create_temporal_pattern_labels(self, patterns_data, account_to_id, prediction_window_days=30):
        """創建基於時間的模式標籤 - 修正帳戶匹配問題"""
        print(f"Creating temporal pattern labels with {prediction_window_days}-day prediction window...")
        
        # 創建標準化的帳戶映射
        normalized_account_to_id = {}
        for (bank_id, account_num), node_id in account_to_id.items():
            normalized_key = self.create_normalized_account_key(bank_id, account_num)
            normalized_account_to_id[normalized_key] = node_id
        
        print(f"Debug: Original account mapping: {len(account_to_id)} accounts")
        print(f"Debug: Normalized account mapping: {len(normalized_account_to_id)} accounts")
        
        # 顯示一些樣本以驗證轉換
        sample_original = list(account_to_id.items())[:3]
        sample_normalized = list(normalized_account_to_id.items())[:3]
        print(f"Debug: Sample original accounts: {sample_original}")
        print(f"Debug: Sample normalized accounts: {sample_normalized}")
        
        num_nodes = len(account_to_id)
        future_risk_labels = torch.zeros(num_nodes, dtype=torch.long)
        pattern_timing_info = {}
        
        total_pattern_accounts = 0
        matched_accounts = 0
        
        # 處理每個洗錢模式
        for pattern_idx, pattern in enumerate(patterns_data):
            transactions = pattern['transactions']
            pattern_type = pattern['pattern_type']
            
            if not transactions:
                continue
            
            # 按時間排序交易
            sorted_transactions = sorted(transactions, key=lambda x: x['TimeStamp'])
            pattern_start_time = sorted_transactions[0]['TimeStamp']
            pattern_end_time = sorted_transactions[-1]['TimeStamp']
            
            # 收集參與此模式的帳戶
            pattern_accounts_in_this_pattern = set()
            for tx in transactions:
                # 使用標準化的帳戶鍵
                from_account_norm = self.create_normalized_account_key(tx['From Bank'], tx['Account'])
                to_account_norm = self.create_normalized_account_key(tx['To Bank'], tx['Account.1'])
                
                total_pattern_accounts += 2  # 每個交易有兩個帳戶
                
                # 嘗試匹配標準化後的帳戶
                if from_account_norm in normalized_account_to_id:
                    pattern_accounts_in_this_pattern.add(from_account_norm)
                    matched_accounts += 1
                elif pattern_idx < 3:  # 只為前3個模式顯示調試
                    print(f"Debug: Pattern {pattern_idx} - from_account {from_account_norm} NOT matched")
                
                if to_account_norm in normalized_account_to_id:
                    pattern_accounts_in_this_pattern.add(to_account_norm)
                    matched_accounts += 1
                elif pattern_idx < 3:
                    print(f"Debug: Pattern {pattern_idx} - to_account {to_account_norm} NOT matched")
            
            # 為匹配到的帳戶設置標籤
            for account_norm in pattern_accounts_in_this_pattern:
                account_id = normalized_account_to_id[account_norm]
                future_risk_labels[account_id] = 1
                
                # 記錄時間信息
                if account_id not in pattern_timing_info:
                    pattern_timing_info[account_id] = {
                        'patterns': [],
                        'earliest_pattern_start': pattern_start_time,
                        'latest_pattern_end': pattern_end_time
                    }
                else:
                    pattern_timing_info[account_id]['earliest_pattern_start'] = min(
                        pattern_timing_info[account_id]['earliest_pattern_start'],
                        pattern_start_time
                    )
                    pattern_timing_info[account_id]['latest_pattern_end'] = max(
                        pattern_timing_info[account_id]['latest_pattern_end'],
                        pattern_end_time
                    )
                
                pattern_timing_info[account_id]['patterns'].append({
                    'pattern_idx': pattern_idx,
                    'pattern_type': pattern_type,
                    'start_time': pattern_start_time,
                    'end_time': pattern_end_time,
                    'prediction_window_start': pattern_start_time - timedelta(days=prediction_window_days)
                })
        
        positive_labels = future_risk_labels.sum().item()
        match_rate = matched_accounts / total_pattern_accounts * 100 if total_pattern_accounts > 0 else 0
        
        print(f"Account matching results:")
        print(f"  Total pattern account references: {total_pattern_accounts}")
        print(f"  Successfully matched: {matched_accounts}")
        print(f"  Match rate: {match_rate:.1f}%")
        print(f"Generated {positive_labels} positive labels from {len(patterns_data)} patterns")
        
        if match_rate < 50:
            print("WARNING: Low match rate. Check account format consistency.")
        else:
            print("✓ Good match rate achieved with normalized account keys")
        
        return future_risk_labels, pattern_timing_info
    
    def create_pattern_features(self, patterns_data, account_to_id):
        """創建模式特徵 - 使用標準化帳戶匹配"""
        print("Creating enhanced pattern features...")
        
        # 創建標準化映射
        normalized_account_to_id = {}
        for (bank_id, account_num), node_id in account_to_id.items():
            normalized_key = self.create_normalized_account_key(bank_id, account_num)
            normalized_account_to_id[normalized_key] = node_id
        
        num_nodes = len(account_to_id)
        feature_dim = len(self.pattern_types) + 10
        pattern_features = torch.zeros(num_nodes, feature_dim)
        
        # 收集參與模式的帳戶
        account_pattern_mapping = defaultdict(list)
        pattern_accounts = set()
        
        for pattern_idx, pattern in enumerate(patterns_data):
            pattern_type = pattern['pattern_type']
            transactions = pattern['transactions']
            
            for tx in transactions:
                from_account_norm = self.create_normalized_account_key(tx['From Bank'], tx['Account'])
                to_account_norm = self.create_normalized_account_key(tx['To Bank'], tx['Account.1'])
                
                if from_account_norm in normalized_account_to_id:
                    pattern_accounts.add(from_account_norm)
                    account_pattern_mapping[from_account_norm].append({
                        'pattern_idx': pattern_idx,
                        'pattern_type': pattern_type,
                        'role': 'sender'
                    })
                
                if to_account_norm in normalized_account_to_id:
                    pattern_accounts.add(to_account_norm)
                    account_pattern_mapping[to_account_norm].append({
                        'pattern_idx': pattern_idx,
                        'pattern_type': pattern_type,
                        'role': 'receiver'
                    })
        
        # 為每個帳戶生成特徵
        for account_norm, node_id in normalized_account_to_id.items():
            if account_norm in pattern_accounts:
                account_patterns = account_pattern_mapping.get(account_norm, [])
                
                # 模式類型one-hot編碼
                pattern_types_involved = set()
                for pattern_info in account_patterns:
                    pattern_type = pattern_info['pattern_type']
                    pattern_features[node_id, pattern_type] = 1.0
                    pattern_types_involved.add(pattern_type)
                
                # 統計特徵
                num_patterns = len(account_patterns)
                num_pattern_types = len(pattern_types_involved)
                sender_count = sum(1 for p in account_patterns if p['role'] == 'sender')
                receiver_count = sum(1 for p in account_patterns if p['role'] == 'receiver')
                
                complexity_scores = {0: 3.0, 1: 3.0, 2: 2.5, 3: 3.0, 4: 3.0, 5: 2.0, 6: 1.0, 7: 0.5}
                total_complexity = sum(complexity_scores[p['pattern_type']] for p in account_patterns)
                avg_complexity = total_complexity / num_patterns if num_patterns > 0 else 0
                
                stat_features = [
                    num_patterns,
                    num_pattern_types,
                    sender_count,
                    receiver_count,
                    sender_count / num_patterns if num_patterns > 0 else 0,
                    receiver_count / num_patterns if num_patterns > 0 else 0,
                    total_complexity,
                    avg_complexity,
                    1.0,
                    min(num_patterns / 5.0, 1.0)
                ]
                
                pattern_features[node_id, len(self.pattern_types):] = torch.tensor(stat_features)
        
        print(f"Pattern features shape: {pattern_features.shape}")
        print(f"Accounts involved in patterns: {len(pattern_accounts)}")
        
        return pattern_features, pattern_accounts
    
    def integrate_patterns_into_data(self, data, account_to_id):
        """將模式信息整合到數據中 - 最終版本"""
        print("=== Integrating Enhanced Pattern Information ===")
        
        patterns_data = self.parse_patterns_file()
        
        if not patterns_data:
            print("No patterns found, returning original data")
            return data
        
        # 創建模式特徵
        pattern_embeddings, pattern_accounts = self.create_pattern_features(patterns_data, account_to_id)
        
        # 創建時間標籤
        enhanced_labels, pattern_timing_info = self.create_temporal_pattern_labels(
            patterns_data, account_to_id, prediction_window_days=30
        )
        
        # 創建模式類型標籤
        pattern_type_labels = self.create_pattern_type_labels(patterns_data, account_to_id)
        
        # 更新data對象
        data.pattern_embeddings = pattern_embeddings
        data.enhanced_labels = enhanced_labels
        data.pattern_type_labels = pattern_type_labels
        data.pattern_timing_info = pattern_timing_info
        
        # 如果原始標籤都是0，替換為增強標籤
        if data.labels.sum().item() == 0:
            print("Original labels are all zeros, replacing with enhanced pattern-based labels")
            data.labels = enhanced_labels
            data.y = enhanced_labels
        
        # 保存統計信息
        pattern_type_counts = {}
        for pattern in patterns_data:
            pattern_name = self._get_pattern_name(pattern['pattern_type'])
            pattern_type_counts[pattern_name] = pattern_type_counts.get(pattern_name, 0) + 1
        
        data.pattern_stats = {
            'num_patterns': len(patterns_data),
            'pattern_types': list(self.pattern_types.keys()),
            'accounts_in_patterns': len(pattern_accounts),
            'pattern_coverage': len(pattern_accounts) / data.num_nodes,
            'positive_label_ratio': enhanced_labels.sum().item() / len(enhanced_labels),
            'pattern_feature_dim': pattern_embeddings.size(1),
            'pattern_type_distribution': pattern_type_counts
        }
        
        print(f"Pattern integration completed successfully:")
        print(f"  - {len(patterns_data)} patterns processed")
        print(f"  - Pattern types: {pattern_type_counts}")
        print(f"  - {len(pattern_accounts)} accounts involved in patterns")
        print(f"  - Pattern coverage: {data.pattern_stats['pattern_coverage']:.2%}")
        print(f"  - Positive label ratio: {data.pattern_stats['positive_label_ratio']:.2%}")
        print(f"  - Pattern embeddings: {data.pattern_embeddings.shape}")
        
        return data
    
    def create_pattern_type_labels(self, patterns_data, account_to_id):
        """創建模式類型多標籤"""
        # 創建標準化映射
        normalized_account_to_id = {}
        for (bank_id, account_num), node_id in account_to_id.items():
            normalized_key = self.create_normalized_account_key(bank_id, account_num)
            normalized_account_to_id[normalized_key] = node_id
        
        num_nodes = len(account_to_id)
        num_pattern_types = len(self.pattern_types)
        pattern_type_labels = torch.zeros(num_nodes, num_pattern_types)
        
        for pattern in patterns_data:
            pattern_type = pattern['pattern_type']
            transactions = pattern['transactions']
            
            for tx in transactions:
                from_account_norm = self.create_normalized_account_key(tx['From Bank'], tx['Account'])
                to_account_norm = self.create_normalized_account_key(tx['To Bank'], tx['Account.1'])
                
                if from_account_norm in normalized_account_to_id:
                    node_id = normalized_account_to_id[from_account_norm]
                    pattern_type_labels[node_id, pattern_type] = 1.0
                
                if to_account_norm in normalized_account_to_id:
                    node_id = normalized_account_to_id[to_account_norm]
                    pattern_type_labels[node_id, pattern_type] = 1.0
        
        return pattern_type_labels