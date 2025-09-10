import torch
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
from data_loader import FraudGTDataLoader
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta

class TemporalSequenceGenerator:
    """
    修正版本的序列生成器，解決時間洩漏問題
    
    核心改進：
    1. 嚴格的時間邊界控制 - 根據每個節點的預測時間窗口裁切序列
    2. 動態時間窗口 - 不同節點可能有不同的預測時間點
    3. 防止未來信息洩漏 - 只使用預測時間點之前的交易歷史
    """
    
    def __init__(self, data_dir, max_sequence_length=32):
        self.data_dir = data_dir
        self.max_sequence_length = max_sequence_length
        
    def analyze_parallel_edges(self, edge_index, edge_attr, edge_timestamps):
        """分析並增強平行邊處理 (DIAM風格)"""
        print("Analyzing parallel edges...")
        
        # 找出平行邊
        edge_pairs = {}
        parallel_edge_stats = defaultdict(list)
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_pair = (src, dst)
            
            if edge_pair not in edge_pairs:
                edge_pairs[edge_pair] = []
            edge_pairs[edge_pair].append(i)
        
        # 統計平行邊
        single_edges = 0
        multi_edges = 0
        max_parallel = 0
        
        for edge_pair, edge_indices in edge_pairs.items():
            if len(edge_indices) == 1:
                single_edges += 1
            else:
                multi_edges += 1
                max_parallel = max(max_parallel, len(edge_indices))
                
                # 計算平行邊統計特徵
                parallel_timestamps = edge_timestamps[edge_indices]
                parallel_attrs = edge_attr[edge_indices]
                
                # 時間跨度
                time_span = parallel_timestamps.max() - parallel_timestamps.min()
                
                # 屬性統計
                attr_stats = {
                    'count': len(edge_indices),
                    'time_span': time_span.item(),
                    'attr_mean': parallel_attrs.mean(dim=0),
                    'attr_std': parallel_attrs.std(dim=0),
                    'attr_sum': parallel_attrs.sum(dim=0)
                }
                
                parallel_edge_stats[edge_pair] = attr_stats
        
        print(f"Single edges: {single_edges}")
        print(f"Multi-edge pairs: {multi_edges}")
        print(f"Max parallel edges: {max_parallel}")
        
        return parallel_edge_stats
    
    def build_temporal_constrained_sequences(self, edge_index, edge_attr, edge_timestamps, 
                                           num_nodes, pattern_timing_info=None):
        """
        核心修正：構建帶時間約束的序列
        
        對每個節點，根據其預測時間窗口，只使用該時間窗口內的交易歷史構建序列
        """
        print("Building temporal-constrained transaction sequences...")
        
        # 分析平行邊
        parallel_stats = self.analyze_parallel_edges(edge_index, edge_attr, edge_timestamps)
        
        # 按時間排序所有邊
        sorted_indices = torch.argsort(edge_timestamps)
        edges_sorted = edge_index[:, sorted_indices]
        edge_attr_sorted = edge_attr[sorted_indices]
        timestamps_sorted = edge_timestamps[sorted_indices]
        
        # 初始化每個節點的序列
        in_sequences = {node_id: [] for node_id in range(num_nodes)}
        out_sequences = {node_id: [] for node_id in range(num_nodes)}
        
        # 獲取基本特徵維度
        base_feature_dim = edge_attr.size(1)
        print(f"Base edge feature dimension: {base_feature_dim}")
        
        # 為每條邊添加到對應節點的序列中
        for eid in range(len(edges_sorted.t())):
            src, dst = edges_sorted[:, eid]
            edge_features = edge_attr_sorted[eid]
            timestamp = timestamps_sorted[eid]
            
            # 確保時間戳特徵處理正確
            if edge_features.size(0) >= base_feature_dim:
                enhanced_features = edge_features
            else:
                enhanced_features = torch.cat([edge_features, timestamp.unsqueeze(0)])
            
            # 添加平行邊統計特徵 (DIAM風格增強)
            edge_pair = (src.item(), dst.item())
            if edge_pair in parallel_stats:
                stats = parallel_stats[edge_pair]
                parallel_features = torch.tensor([
                    float(stats['count']),
                    float(stats['time_span']),
                ], dtype=torch.float)
            else:
                parallel_features = torch.tensor([1.0, 0.0], dtype=torch.float)
            
            # 組合所有特徵
            final_features = torch.cat([enhanced_features, parallel_features])
            
            # 添加時間戳信息用於後續的時間過濾
            transaction_info = {
                'features': final_features,
                'timestamp': timestamp.item(),
                'edge_id': eid
            }
            
            # 出邊序列（從該節點發出的交易）
            out_sequences[src.item()].append(transaction_info)
            # 入邊序列（進入該節點的交易）
            in_sequences[dst.item()].append(transaction_info)
        
        # 應用時間約束過濾
        filtered_in_sequences, filtered_out_sequences = self.apply_temporal_constraints(
            in_sequences, out_sequences, pattern_timing_info, num_nodes
        )
        
        return filtered_in_sequences, filtered_out_sequences
    
    def apply_temporal_constraints(self, in_sequences, out_sequences, pattern_timing_info, num_nodes):
        """
        關鍵修正：應用時間約束過濾
        
        對每個節點，根據其預測時間窗口，只保留該時間窗口之前的交易
        """
        print("Applying temporal constraints to prevent data leakage...")
        
        filtered_in_sequences = {}
        filtered_out_sequences = {}
        
        nodes_with_constraints = 0
        nodes_without_constraints = 0
        
        for node_id in range(num_nodes):
            # 獲取該節點的時間約束
            if pattern_timing_info and node_id in pattern_timing_info:
                # 該節點參與洗錢模式，使用模式的預測時間窗口
                timing_info = pattern_timing_info[node_id]
                
                # 修正：使用正確的鍵名來獲取時間信息
                try:
                    # 使用最早的預測窗口開始時間作為截止點
                    if 'patterns' in timing_info:
                        cutoff_times = []
                        for p in timing_info['patterns']:
                            if 'prediction_window_start' in p:
                                cutoff_times.append(p['prediction_window_start'])
                            elif 'pattern_start' in p:
                                # 使用模式開始時間減去預測窗口
                                from datetime import timedelta
                                pattern_start = p['pattern_start']
                                cutoff_time = pattern_start - timedelta(days=30)
                                cutoff_times.append(cutoff_time)
                        
                        if cutoff_times:
                            cutoff_time = min(cutoff_times)
                            cutoff_timestamp = cutoff_time.timestamp() / (24 * 3600) if hasattr(cutoff_time, 'timestamp') else cutoff_time
                        else:
                            cutoff_timestamp = float('inf')
                    else:
                        # 如果沒有 patterns 鍵，使用 earliest_pattern_start
                        if 'earliest_pattern_start' in timing_info:
                            from datetime import timedelta
                            cutoff_time = timing_info['earliest_pattern_start'] - timedelta(days=30)
                            cutoff_timestamp = cutoff_time.timestamp() / (24 * 3600)
                        else:
                            cutoff_timestamp = float('inf')
                            
                    nodes_with_constraints += 1
                    
                except Exception as e:
                    print(f"Warning: Error processing timing info for node {node_id}: {e}")
                    cutoff_timestamp = float('inf')
                    nodes_without_constraints += 1
            else:
                # 該節點沒有特定時間約束，使用所有歷史交易
                cutoff_timestamp = float('inf')
                nodes_without_constraints += 1
            
            # 過濾入邊序列
            node_in_sequence = in_sequences.get(node_id, [])
            filtered_in = []
            for transaction_info in node_in_sequence:
                if transaction_info['timestamp'] <= cutoff_timestamp:
                    filtered_in.append(transaction_info['features'])
            filtered_in_sequences[node_id] = filtered_in
            
            # 過濾出邊序列
            node_out_sequence = out_sequences.get(node_id, [])
            filtered_out = []
            for transaction_info in node_out_sequence:
                if transaction_info['timestamp'] <= cutoff_timestamp:
                    filtered_out.append(transaction_info['features'])
            filtered_out_sequences[node_id] = filtered_out
        
        print(f"Temporal filtering applied:")
        print(f"  節點有時間約束: {nodes_with_constraints}")
        print(f"  節點無時間約束: {nodes_without_constraints}")
        
        # 統計過濾效果
        total_transactions_before = sum(len(seq) for seq in in_sequences.values()) + sum(len(seq) for seq in out_sequences.values())
        total_transactions_after = sum(len(seq) for seq in filtered_in_sequences.values()) + sum(len(seq) for seq in filtered_out_sequences.values())
        
        if total_transactions_before > 0:
            retention_rate = total_transactions_after / total_transactions_before
            print(f"  交易保留率: {retention_rate:.2%}")
        
        return filtered_in_sequences, filtered_out_sequences
    
    def truncate_and_pad_sequences(self, sequences, max_length):
        """
        截斷和填充序列到固定長度
        修正版本：動態獲取特徵維度
        """
        processed_sequences = []
        sequence_lengths = []
        
        # 動態獲取特徵維度
        feature_dim = None
        for node_id in range(len(sequences)):
            if len(sequences[node_id]) > 0:
                feature_dim = sequences[node_id][0].size(0)
                break
        
        if feature_dim is None:
            # 如果所有節點都沒有交易，使用默認維度
            # 基本特徵維度 + 時間戳 + 平行邊統計(2)
            feature_dim = 10  # 這個需要根據實際的edge_attr維度調整
            print(f"警告: 未找到交易，使用預設特徵維度={feature_dim}")
        
        print(f"使用特徵維度: {feature_dim}")
        
        for node_id in range(len(sequences)):
            node_sequence = sequences[node_id]
            
            if len(node_sequence) == 0:
                # 如果節點沒有交易，創建零填充
                padded_seq = torch.zeros(1, feature_dim)
                processed_sequences.append(padded_seq)
                sequence_lengths.append(1)
                
            else:
                # 將列表轉換為張量
                node_tensor = torch.stack(node_sequence)
                
                # 截斷到最大長度（保留最近的交易）
                if len(node_tensor) > max_length:
                    node_tensor = node_tensor[-max_length:]
                
                processed_sequences.append(node_tensor)
                sequence_lengths.append(len(node_tensor))
        
        return processed_sequences, torch.tensor(sequence_lengths)
    
    def generate_temporal_sequences(self, data):
        """
        核心方法：生成帶時間約束的序列
        """
        print(f"生成時間約束序列，最大長度 {self.max_sequence_length}...")
        
        # 從data中獲取模式時間信息
        pattern_timing_info = getattr(data, 'pattern_timing_info', None)
        
        if pattern_timing_info:
            print(f"找到 {len(pattern_timing_info)} 個節點的時間約束資訊")
        else:
            print("未找到模式時間資訊，將使用所有歷史交易")
        
        # 構建帶時間約束的節點序列
        in_sequences, out_sequences = self.build_temporal_constrained_sequences(
            data.edge_index, 
            data.edge_attr, 
            data.edge_timestamps, 
            data.num_nodes,
            pattern_timing_info
        )
        
        # 處理入邊序列
        print("處理時間約束的入邊序列...")
        in_sequences_list = [in_sequences[i] for i in range(data.num_nodes)]
        processed_in_sequences, in_lengths = self.truncate_and_pad_sequences(
            in_sequences_list, self.max_sequence_length
        )
        
        # 處理出邊序列
        print("處理時間約束的出邊序列...")
        out_sequences_list = [out_sequences[i] for i in range(data.num_nodes)]
        processed_out_sequences, out_lengths = self.truncate_and_pad_sequences(
            out_sequences_list, self.max_sequence_length
        )
        
        # 轉換為numpy數組
        in_sequences_np = np.array(processed_in_sequences, dtype=object)
        out_sequences_np = np.array(processed_out_sequences, dtype=object)
        
        print(f"已為 {data.num_nodes} 個節點生成時間約束序列")
        print(f"平均入邊序列長度: {in_lengths.float().mean():.2f}")
        print(f"平均出邊序列長度: {out_lengths.float().mean():.2f}")
        
        # 特徵維度檢查
        if len(processed_in_sequences) > 0:
            feature_dim = processed_in_sequences[0].size(-1)
            print(f"時間約束特徵維度: {feature_dim}")
        
        return {
            'in_sequences': in_sequences_np,
            'out_sequences': out_sequences_np,
            'in_lengths': in_lengths,
            'out_lengths': out_lengths
        }
    
    def save_sequences(self, sequences_data, suffix=''):
        """保存序列資料"""
        length = self.max_sequence_length
        
        # 定義文件路徑
        in_seq_path = os.path.join(self.data_dir, f'temporal_in_sentences_{length}{suffix}.npy')
        out_seq_path = os.path.join(self.data_dir, f'temporal_out_sentences_{length}{suffix}.npy')
        in_len_path = os.path.join(self.data_dir, f'temporal_in_sentences_len_{length}{suffix}.pt')
        out_len_path = os.path.join(self.data_dir, f'temporal_out_sentences_len_{length}{suffix}.pt')
        
        # 保存資料
        np.save(in_seq_path, sequences_data['in_sequences'])
        np.save(out_seq_path, sequences_data['out_sequences'])
        torch.save(sequences_data['in_lengths'], in_len_path)
        torch.save(sequences_data['out_lengths'], out_len_path)
        
        print(f"時間約束序列已保存至:")
        print(f"  {in_seq_path}")
        print(f"  {out_seq_path}")
        print(f"  {in_len_path}")
        print(f"  {out_len_path}")
    
    def load_sequences(self, suffix='', use_temporal=True):
        """載入序列資料"""
        length = self.max_sequence_length
        
        if use_temporal:
            prefix = 'temporal_'
        else:
            prefix = 'enhanced_'  # 回退到增強版本
        
        in_seq_path = os.path.join(self.data_dir, f'{prefix}in_sentences_{length}{suffix}.npy')
        out_seq_path = os.path.join(self.data_dir, f'{prefix}out_sentences_{length}{suffix}.npy')
        in_len_path = os.path.join(self.data_dir, f'{prefix}in_sentences_len_{length}{suffix}.pt')
        out_len_path = os.path.join(self.data_dir, f'{prefix}out_sentences_len_{length}{suffix}.pt')
        
        # 檢查時間約束版本是否存在
        if use_temporal and not all(os.path.exists(p) for p in [in_seq_path, out_seq_path, in_len_path, out_len_path]):
            print("時間約束序列不存在，回退到增強序列...")
            return self.load_sequences(suffix, use_temporal=False)
        
        # 檢查基本版本是否存在
        if not use_temporal:
            for path in [in_seq_path, out_seq_path, in_len_path, out_len_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"序列檔案不存在: {path}")
        
        # 載入資料
        in_sequences = np.load(in_seq_path, allow_pickle=True)
        out_sequences = np.load(out_seq_path, allow_pickle=True)
        in_lengths = torch.load(in_len_path)
        out_lengths = torch.load(out_len_path)
        
        print(f"已載入 {'時間約束' if use_temporal else '增強'} 序列")
        print(f"  入邊序列形狀: {in_sequences.shape}")
        print(f"  出邊序列形狀: {out_sequences.shape}")
        print(f"  平均序列長度: 入邊={in_lengths.float().mean():.1f}, 出邊={out_lengths.float().mean():.1f}")
        
        return {
            'in_sequences': in_sequences,
            'out_sequences': out_sequences,
            'in_lengths': in_lengths,
            'out_lengths': out_lengths
        }
    
    def prepare_for_training(self, sequences_data):
        """準備訓練所需的張量格式"""
        # 轉換為pad_sequence格式
        in_sequences = pad_sequence([torch.tensor(seq) for seq in sequences_data['in_sequences']], batch_first=True)
        out_sequences = pad_sequence([torch.tensor(seq) for seq in sequences_data['out_sequences']], batch_first=True)
        
        print(f"訓練序列已準備:")
        print(f"  入邊序列: {in_sequences.shape}")
        print(f"  出邊序列: {out_sequences.shape}")
        print(f"  特徵維度: {in_sequences.shape[-1]}")
        
        return {
            'in_sequences': in_sequences,
            'out_sequences': out_sequences,
            'in_lengths': sequences_data['in_lengths'],
            'out_lengths': sequences_data['out_lengths'],
            'rnn_input_dim': in_sequences.size(-1)
        }

# 為了向後兼容，保留原始類別名稱
class EnhancedSequenceGenerator(TemporalSequenceGenerator):
    """向後兼容的別名"""
    
    def generate_sequences(self, data):
        """向後兼容方法"""
        return self.generate_temporal_sequences(data)

def generate_fraudgt_sequences(data_dir, max_length=32, prediction_window=30, use_temporal=True):
    """
    完整的時間約束序列生成流程
    """
    print("=== 時間約束 FraudGT 序列生成 ===")
    print("特性: 時間洩漏防護 + 多邊分析 + 動態特徵維度")
    
    # 初始化載入器和生成器
    data_loader = FraudGTDataLoader(data_dir, prediction_window)
    sequence_generator = TemporalSequenceGenerator(data_dir, max_length)
    
    # 檢查是否已有轉換後的資料
    data_file = os.path.join(data_dir, 'data.pt')
    
    if os.path.exists(data_file):
        print("載入現有圖資料...")
        data = data_loader.load_data('data.pt')
    else:
        print("將CSV轉換為圖格式...")
        data = data_loader.convert_to_diam_format()
        data_loader.save_data(data, 'data.pt')
    
    # 生成時間約束序列
    print("\n生成時間約束交易序列...")
    sequences_data = sequence_generator.generate_temporal_sequences(data)
    
    # 保存序列
    print("\n保存時間約束序列...")
    sequence_generator.save_sequences(sequences_data)
    
    # 驗證生成的序列
    print(f"\n=== 序列生成摘要 ===")
    print(f"總節點數: {data.num_nodes}")
    print(f"有交易的節點: {sum(1 for i in range(len(sequences_data['in_sequences'])) if len(sequences_data['in_sequences'][i]) > 0)}")
    print(f"平均序列長度: {sequences_data['in_lengths'].float().mean():.2f}")
    print(f"最大序列長度: {sequences_data['in_lengths'].max().item()}")
    print(f"特徵維度: {sequence_generator.prepare_for_training(sequences_data)['rnn_input_dim']}")
    
    # 檢查時間約束效果
    if hasattr(data, 'pattern_timing_info') and data.pattern_timing_info:
        print(f"時間約束節點: {len(data.pattern_timing_info)}")
        print("時間洩漏防護: 已啟用")
    else:
        print("時間約束節點: 0")
        print("時間洩漏防護: 未使用（無模式時間資訊）")
    
    print("\n=== 時間約束序列生成完成 ===")
    return data, sequences_data