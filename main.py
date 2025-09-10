import pickle
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

# 導入自定義模組
from data_loader import FraudGTDataLoader
from sequence_generator import EnhancedSequenceGenerator
from model import Binary_Classifier
from layers import *
from utils import *

def set_random_seed(seed):
    """設置隨機種子以確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_graph_data_adapter(data, train_sequences, val_sequences, test_sequences, lens_in, lens_out):
    """創建適配器，將FraudGT資料格式轉換為DIAM期望的格式"""
    
    class GraphDataAdapter:
        def __init__(self):
            self.labels = data.labels
            self.edge_index = data.edge_index
            self.num_nodes = data.num_nodes
            self.lens_in = lens_in
            self.lens_out = lens_out
            
            # 創建mask
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
            self.test_mask = data.test_mask
            self.train_label = data.train_label
            self.val_label = data.val_label
            self.test_label = data.test_label
            
            # 添加額外資訊
            self.pattern_info = getattr(data, 'pattern_stats', None)
            
        def get_subgraph(self, mask, relabel_nodes=True):
            """獲取子圖"""
            class SubGraph:
                def __init__(self):
                    pass
            
            sub_g = SubGraph()
            sub_g.labels = self.labels[mask]
            sub_g.train_label = self.train_label[mask] if hasattr(self, 'train_label') else torch.ones_like(self.labels[mask], dtype=torch.bool)
            sub_g.lens_in = self.lens_in[mask]
            sub_g.lens_out = self.lens_out[mask]
            
            if relabel_nodes:
                # 重新映射邊索引
                node_mapping = torch.zeros(self.num_nodes, dtype=torch.long)
                node_mapping[mask] = torch.arange(mask.sum())
                
                # 篩選邊
                edge_mask = mask[self.edge_index[0]] & mask[self.edge_index[1]]
                sub_g.edge_index = node_mapping[self.edge_index[:, edge_mask]]
                sub_g.num_nodes = mask.sum().item()
            else:
                sub_g.edge_index = self.edge_index
                sub_g.num_nodes = self.num_nodes
                
            return sub_g
    
    return GraphDataAdapter()

def load_fraudgt_data(args):
    """載入FraudGT資料並轉換為DIAM格式"""
    data_dir = f'./data/{args.data}'
    
    print("=== Loading FraudGT Data ===")
    
    # 1. 載入圖資料
    data_loader = FraudGTDataLoader(data_dir, prediction_window_days=args.prediction_window)
    
    try:
        data = data_loader.load_data('data.pt')
        print("✓ Loaded existing graph data")
    except FileNotFoundError:
        print("Converting CSV to graph format...")
        data = data_loader.convert_to_diam_format()
        data_loader.save_data(data, 'data.pt')
        print("✓ Graph data converted and saved")
    
    # 2. 載入序列資料
    sequence_generator = EnhancedSequenceGenerator(data_dir, args.length)
    
    try:
        sequences_data = sequence_generator.load_sequences() 
        print("✓ Loaded existing enhanced sequences")
    except FileNotFoundError:
        print("Generating enhanced sequences...")
        sequences_data = sequence_generator.generate_sequences(data)
        sequence_generator.save_sequences(sequences_data)
        print("✓ Enhanced sequences generated and saved")
    
    # 3. 準備訓練格式
    training_sequences = sequence_generator.prepare_for_training(sequences_data)
    
    in_sentences = training_sequences['in_sequences']
    out_sentences = training_sequences['out_sequences']
    lens_in = training_sequences['in_lengths']
    lens_out = training_sequences['out_lengths']
    rnn_input_dim = training_sequences['rnn_input_dim']
    
    print(f"✓ Data loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"✓ RNN input dimension: {rnn_input_dim}")
    print(f"✓ Positive labels: {data.labels.sum()}/{len(data.labels)} ({data.labels.sum()/len(data.labels)*100:.1f}%)")
    
    # 4. 創建適配的圖資料結構
    g = create_graph_data_adapter(data, in_sentences, out_sentences, in_sentences, lens_in, lens_out)
    
    return g, in_sentences, out_sentences, rnn_input_dim

def create_data_loaders(g, args):
    """創建資料載入器"""
    print("=== Creating Data Loaders ===")
    
    # 創建訓練子圖
    g_train = g.get_subgraph(g.train_mask, relabel_nodes=True)
    
    # 獲取節點索引
    train_nid = torch.nonzero(g_train.train_label, as_tuple=True)[0]
    val_nid = torch.nonzero(g.val_label, as_tuple=True)[0]
    test_nid = torch.nonzero(g.test_label, as_tuple=True)[0]
    
    print(f"Train/Val/Test nodes before processing: {len(train_nid)}/{len(val_nid)}/{len(test_nid)}")
    
    # 處理類別不平衡
    if args.oversample and args.oversample > 0:
        try:
            from imblearn.over_sampling import RandomOverSampler
            oversample = RandomOverSampler(sampling_strategy=args.oversample, random_state=args.random_state)
            nid_resampled, _ = oversample.fit_resample(
                train_nid.reshape(-1, 1), 
                g_train.labels[g_train.train_label]
            )
            train_nid = torch.as_tensor(nid_resampled.reshape(-1))
            print(f"✓ Oversampled training nodes: {len(train_nid)}")
        except ImportError:
            print("⚠ imbalanced-learn not installed, skipping oversampling")
    
    # 創建採樣器
    if args.undersample:
        sampler = BalancedSampler(g_train.labels[train_nid])
        print("✓ Using balanced sampler")
    else:
        sampler = None
    
    # 創建DataLoader
    loader_train = DualNeighborSampler(
        g_train.edge_index,
        sizes=[25, 10],
        node_idx=train_nid,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=None if sampler else True,
    )
    
    loader_val = DualNeighborSampler(
        g.edge_index,
        node_idx=val_nid,
        sizes=[25, 10],
        batch_size=int(args.batch_size/2),
    )
    
    loader_test = DualNeighborSampler(
        g.edge_index,
        node_idx=test_nid,
        sizes=[25, 10],
        batch_size=int(args.batch_size/2),
    )
    
    print(f"✓ Data loaders created")
    print(f"  Train batches: {len(loader_train)}")
    print(f"  Val batches: {len(loader_val)}")  
    print(f"  Test batches: {len(loader_test)}")
    
    return g_train, loader_train, loader_val, loader_test

def create_model_and_optimizer(args, rnn_input_dim, g_train, device):
    """創建模型和優化器"""
    print("=== Creating Model and Optimizer ===")
    
    # 創建模型
    model = Binary_Classifier(
        in_channels=0,  # 不使用節點特徵
        hidden_channels=args.num_hidden,
        out_channels=args.num_outputs,
        rnn_in_channels=rnn_input_dim,
        num_layers=args.num_layers,
        rnn_agg=args.rnn_agg,
        encoder_layer=args.model,
        concat_feature=args.concat_feature,
        dropout=args.dropout,
        emb_first=args.emb_first,
        gnn_norm=args.gnn_norm,
        lstm_norm=args.lstm_norm,
        graph_op=args.graph_op,
        decoder_layers=args.decoder_layers,
        aggr=args.aggr
    ).to(device)
    
    # 設置損失函數
    if args.reweight:
        weights = torch.FloatTensor([
            1/(g_train.labels==0).sum().item(),
            1/(g_train.labels==1).sum().item()
        ]).to(device)
        print('✓ Using reweighted loss')
        loss_fcn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fcn = nn.CrossEntropyLoss()
    
    # 設置優化器
    weight_params = []
    for pname, p in model.encoder.named_parameters():
        if 'proj' in pname or 'adp' in pname:
            weight_params += [p]
    
    all_params = model.parameters()
    params_id = list(map(id, weight_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    
    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.lr}, 
        {'params': weight_params, 'lr': args.weight_lr}
    ])
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.num_epochs//3, args.num_epochs*2//3], gamma=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  RNN input dim: {rnn_input_dim}")
    print(f"  Hidden dim: {args.num_hidden}")
    
    return model, loss_fcn, optimizer, scheduler

def train_and_evaluate(model, g, g_train, loaders, loss_fcn, optimizer, scheduler, 
                      in_sentences, out_sentences, args, device):
    """訓練和評估模型"""
    loader_train, loader_val, loader_test = loaders
    
    print("=== Starting Training ===")
    
    # 準備序列選擇器
    in_sentences_train = in_sentences[g.train_mask]
    out_sentences_train = out_sentences[g.train_mask]
    sens_selector_train = PreSentences_light(train=True, train_mask=g.train_mask)
    sens_selector = PreSentences_light()
    
    # 記錄訓練歷史
    best_val_f1 = 0.0
    best_test_results = None
    patience_counter = 0
    training_history = []
    
    log_every_epoch = max(1, args.num_epochs // 20)  # 至少每5%的epochs記錄一次
    
    for epoch in range(args.num_epochs):
        # 訓練階段
        model.train()
        epoch_loss = 0
        epoch_batches = 0
        attention_weights = []
        
        for batch_idx, (sub_graph, subset, batch_size) in enumerate(loader_train):
            sub_graph = sub_graph.to(device)
            
            # 獲取序列
            in_pack, lens_in = sens_selector_train.select(subset, in_sentences_train, g_train.lens_in)
            out_pack, lens_out = sens_selector_train.select(subset, out_sentences_train, g_train.lens_out)
            
            in_pack = in_pack.to(device)
            out_pack = out_pack.to(device)
            
            # 前向傳播
            optimizer.zero_grad()
            batch_pred, attention = model(in_pack, out_pack, lens_in, lens_out, sub_graph)
            
            # 計算損失
            loss = loss_fcn(batch_pred[:batch_size], g_train.labels[subset][:batch_size].to(device))
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            attention_weights.append(attention.detach().cpu())
            
            if batch_idx % max(len(loader_train) // 5, 1) == 0:
                print(f'Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx+1}/{len(loader_train)}, '
                      f'Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / epoch_batches
        
        # 記錄注意力權重統計
        if attention_weights:
            all_attention = torch.cat(attention_weights, 0)
            attention_mean = all_attention.mean(0)
            attention_std = all_attention.std(0)
        
        # 評估階段
        if (epoch + 1) % log_every_epoch == 0 or epoch == args.num_epochs - 1:
            print(f"\n--- Epoch {epoch+1} Evaluation ---")
            
            val_results, test_results = evaluate_light(
                model, g, loader_val, loader_test, sens_selector,
                in_sentences, out_sentences, device
            )
            
            # 打印結果
            print(f'Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | '
                  f'Val F1: {val_results["F1_val"]:.4f} | Val AUC: {val_results["AUC_val"]:.4f} | '
                  f'Test F1: {test_results["F1_test"]:.4f} | Test AUC: {test_results["AUC_test"]:.4f}')
            
            # 記錄歷史
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'val_f1': val_results["F1_val"],
                'val_auc': val_results["AUC_val"],
                'test_f1': test_results["F1_test"],
                'test_auc': test_results["AUC_test"],
                'lr': optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_stats)
            
            # 早停和最佳模型保存
            if val_results['F1_val'] > best_val_f1:
                best_val_f1 = val_results['F1_val']
                best_test_results = test_results.copy()
                patience_counter = 0
                
                # 保存最佳模型
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_f1': best_val_f1,
                    'test_results': best_test_results,
                    'training_history': training_history,
                    'args': args,
                    'model_config': {
                        'in_channels': 0,
                        'hidden_channels': args.num_hidden,
                        'out_channels': args.num_outputs,
                        'rnn_in_channels': model.lstm_in.input_size if hasattr(model, 'lstm_in') else None,
                        'num_layers': args.num_layers,
                        'rnn_agg': args.rnn_agg,
                        'encoder_layer': args.model,
                        'concat_feature': args.concat_feature,
                        'dropout': args.dropout,
                        'emb_first': args.emb_first,
                        'gnn_norm': args.gnn_norm,
                        'lstm_norm': args.lstm_norm,
                        'graph_op': args.graph_op,
                        'decoder_layers': args.decoder_layers,
                        'aggr': args.aggr
                    }
                }
                
                model_save_path = os.path.join(f'./data/{args.data}', 'best_model.pt')
                torch.save(checkpoint, model_save_path)
                print(f'✓ Best model saved (Val F1: {best_val_f1:.4f})')
                
            else:
                patience_counter += 1
                
            # 早停檢查
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at epoch {epoch+1} (patience: {args.patience})')
                break
                
            print()  # 空行分隔
    
    print("=== Training Completed ===")
    print(f'Best Validation F1: {best_val_f1:.4f}')
    
    if best_test_results:
        print('Final Test Results (at best validation):')
        for metric, value in best_test_results.items():
            print(f'  {metric}: {value:.4f}')
    
    return best_test_results, training_history

def save_results(results, training_history, args):
    """保存訓練結果"""
    results_dir = f'./data/{args.data}/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存最終結果
    final_results = {
        'test_results': results,
        'training_history': training_history,
        'args': vars(args),
        'model_info': {
            'architecture': args.model,
            'hidden_dim': args.num_hidden,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    }
    
    results_file = os.path.join(results_dir, f'results_{args.model}_{args.random_state}.pt')
    torch.save(final_results, results_file)
    print(f'✓ Results saved to {results_file}')

def main(args):
    """主函數"""
    print("=" * 60)
    print("FraudGT Account Risk Prediction with DIAM")
    print("=" * 60)
    
    # 設置設備和隨機種子
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"✓ Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    set_random_seed(args.random_state)
    print(f"✓ Random seed set to {args.random_state}")
    
    # 載入資料
    g, in_sentences, out_sentences, rnn_input_dim = load_fraudgt_data(args)
    
    # 創建資料載入器
    g_train, loader_train, loader_val, loader_test = create_data_loaders(g, args)
    loaders = (loader_train, loader_val, loader_test)
    
    # 創建模型
    model, loss_fcn, optimizer, scheduler = create_model_and_optimizer(
        args, rnn_input_dim, g_train, device
    )
    
    # 訓練和評估
    results, history = train_and_evaluate(
        model, g, g_train, loaders, loss_fcn, optimizer, scheduler,
        in_sentences, out_sentences, args, device
    )
    
    # 保存結果
    save_results(results, history, args)
    
    return results

if __name__ == '__main__':
    # 確保必要目錄存在
    needed_dirs = ['./results', './embs', './model', './profilers', './temp']
    for dir_name in needed_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # 參數解析器
    argparser = argparse.ArgumentParser(description='FraudGT Account Risk Prediction')
    
    # 資料相關參數
    argparser.add_argument('--data', type=str, default='HI-Small',
                          help='Dataset name')
    argparser.add_argument('--prediction_window', type=int, default=30,
                          help='Prediction window in days')
    argparser.add_argument('--length', type=int, default=32,
                          help='Maximum sequence length')
    
    # 模型相關參數
    argparser.add_argument('--model', type=str, default='dualcata-tanh-16',
                          help='GNN model type')
    argparser.add_argument('--num_hidden', type=int, default=128,
                          help='Hidden layer dimension')
    argparser.add_argument('--num_layers', type=int, default=2,
                          help='Number of GNN layers')
    argparser.add_argument('--num_outputs', type=int, default=2,
                          help='Number of output classes')
    argparser.add_argument('--decoder_layers', type=int, default=2,
                          help='Number of decoder layers')
    argparser.add_argument('--rnn_agg', type=str, default='max',
                          help='RNN aggregation method {last, max, min, sum, mean}')
    argparser.add_argument('--lstm_norm', type=str, default='ln',
                          help='LSTM normalization {ln, bn, none}')
    argparser.add_argument('--gnn_norm', type=str, default='bn',
                          help='GNN normalization {ln, bn, none}')
    argparser.add_argument('--graph_op', type=str, default='',
                          help='Graph operations')
    argparser.add_argument('--aggr', type=str, default='add',
                          help='Message aggregation method')
    
    # 訓練相關參數
    argparser.add_argument('--num_epochs', type=int, default=100,
                          help='Number of training epochs')
    argparser.add_argument('--batch_size', type=int, default=128,
                          help='Batch size')
    argparser.add_argument('--lr', type=float, default=0.001,
                          help='Learning rate')
    argparser.add_argument('--weight_lr', type=float, default=0.001,
                          help='Weight learning rate for attention')
    argparser.add_argument('--dropout', type=float, default=0.2,
                          help='Dropout rate')
    argparser.add_argument('--patience', type=int, default=20,
                          help='Early stopping patience')
    
    # 資料處理參數
    argparser.add_argument('--reweight', action='store_true',
                          help='Use class reweighting')
    argparser.add_argument('--undersample', action='store_true',
                          help='Use balanced undersampling')
    argparser.add_argument('--oversample', type=float, default=0.0,
                          help='Oversampling ratio (0 to disable)')
    
    # 其他參數
    argparser.add_argument('--gpu', type=int, default=0,
                          help='GPU device ID (-1 for CPU)')
    argparser.add_argument('--random_state', type=int, default=42,
                          help='Random seed')
    argparser.add_argument('--emb_first', type=int, default=1,
                          help='Whether to embed RNN input first')
    argparser.add_argument('--concat_feature', type=int, default=0,
                          help='Concatenate features')
    
    # 解析參數並設置默認值
    args = argparser.parse_args()
    
    # 運行主程式
    try:
        results = main(args)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()