import pickle
import os
from torch_geometric.nn import LabelPropagation
import cProfile, pstats
import os
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import torch
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
import random
from torch.autograd import Variable


from model import *
from layers import *
from utils import *
from dataloader import *


def create_temporal_subgraph(data, edge_mask):
    """
    根據邊的 mask 創建時間子圖（Transductive: 保留所有節點）
    
    Args:
        data: 完整的圖資料
        edge_mask: 哪些邊在當前時間階段可用
    
    Returns:
        temporal_data: 包含所有節點但只有部分邊的圖
    """
    temporal_data = Data()
    
    # 保留所有節點的資料
    num_nodes = data.y.shape[0]
    temporal_data.num_nodes = num_nodes
    temporal_data.y = data.y
    temporal_data.labels = data.labels
    temporal_data.train_mask = data.train_mask
    temporal_data.val_mask = data.val_mask
    temporal_data.test_mask = data.test_mask
    temporal_data.train_label = data.train_label
    temporal_data.val_label = data.val_label
    temporal_data.test_label = data.test_label
    
    # 只保留當前時間階段可用的邊
    temporal_data.edge_index = data.edge_index[:, edge_mask]
    temporal_data.edge_attr = data.edge_attr[edge_mask]
    print("根據邊的 mask 創建時間子圖")
    
    return temporal_data

def evaluate_light_with_sequences(model, g, loader, sens_selector, in_sentences, out_sentences, device='cpu', phase='val'):
    """
    使用特定時間過濾後的序列進行評估
    """
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        preds = []
        labels = []
        for loader_id, (sub_graph, subset, batch_size) in enumerate(loader):
            sub_graph = sub_graph.to(device)
            in_pack, lens_in = sens_selector.select(subset, in_sentences, g.lens_in)
            out_pack, lens_out = sens_selector.select(subset, out_sentences, g.lens_out)
            in_pack = in_pack.to(device)
            out_pack = out_pack.to(device)
            batch_pred, *_ = model(in_pack, out_pack, lens_in, lens_out, sub_graph)
            preds.append(batch_pred.cpu()[:batch_size])
            labels.append(g.labels.cpu()[subset][:batch_size])
        
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        results = calculate_metrics(labels, preds)
        
        return {
            f'Precision_{phase}': results[0], 
            f'Recall_{phase}': results[1], 
            f'F1_{phase}': results[2], 
            f'AUC_{phase}': results[3]
        }

def main(args):
    # Load data
    data_path = os.path.join(data_dir, 'data.pt')
    print(data_path)
    
    # 使用嚴謹的 Transductive temporal split
    g, n_classes = load_data(data_path, train_rate=args.train_rate, 
                            anomaly_rate=args.anomaly_rate, random_state=args.random_state,
                            temporal_split_mode=True)
    
    length = args.length
    
    # 載入完整的序列（包含所有時間）
    print(f"\n{'='*70}")
    print("載入原始交易序列...")
    print(f"{'='*70}")
    in_sentences_full = np.load(os.path.join(data_dir, f'in_sentences_{length}.npy'), allow_pickle=True)
    out_sentences_full = np.load(os.path.join(data_dir, f'out_sentences_{length}.npy'), allow_pickle=True)
    in_sentences_len_full = torch.load(os.path.join(data_dir, f'in_sentences_len_{length}.pt'))
    out_sentences_len_full = torch.load(os.path.join(data_dir, f'out_sentences_len_{length}.pt'))
    
    # 轉換為 tensor
    in_sentences_full = pad_sequence(in_sentences_full.tolist(), batch_first=True)
    out_sentences_full = pad_sequence(out_sentences_full.tolist(), batch_first=True)
    
    print(f"原始序列形狀: {in_sentences_full.shape}")
    
    # ===== 嚴格的 Transductive 時間過濾：根據不同階段過濾序列 =====
    print(f"\n{'='*70}")
    print("根據時間閾值過濾交易序列...")
    print(f"{'='*70}")
    
    # 訓練序列：只保留 t < t1 的交易
    in_sentences_train, out_sentences_train, in_lens_train, out_lens_train = filter_sequences_by_time(
        in_sentences_full, out_sentences_full, 
        in_sentences_len_full, out_sentences_len_full,
        time_threshold=g.train_time_threshold,
        sort_dim=4,  # 時間戳在最後一維(加了GFP後，timestamp在第四維)
        max_length=length
    )
    print(f"訓練序列 (t < {g.train_time_threshold:.2f}): {in_sentences_train.shape}")
    print(f"  平均序列長度: In={in_lens_train.float().mean():.2f}, Out={out_lens_train.float().mean():.2f}")
    
    # 驗證序列：只保留 t < t2 的交易
    in_sentences_val, out_sentences_val, in_lens_val, out_lens_val = filter_sequences_by_time(
        in_sentences_full, out_sentences_full,
        in_sentences_len_full, out_sentences_len_full,
        time_threshold=g.val_time_threshold,
        sort_dim=-1,
        max_length=length
    )
    print(f"驗證序列 (t < {g.val_time_threshold:.2f}): {in_sentences_val.shape}")
    print(f"  平均序列長度: In={in_lens_val.float().mean():.2f}, Out={out_lens_val.float().mean():.2f}")
    
    # 測試序列：使用所有交易
    in_sentences_test = in_sentences_full
    out_sentences_test = out_sentences_full
    in_lens_test = in_sentences_len_full
    out_lens_test = out_sentences_len_full
    print(f"測試序列 (所有時間): {in_sentences_test.shape}")
    print(f"  平均序列長度: In={in_lens_test.float().mean():.2f}, Out={out_lens_test.float().mean():.2f}")
    print(f"{'='*70}\n")
    
    rnn_in_channels = out_sentences_full.size(-1)
    
    # full graph training
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    n_classes = 2
    
    g.num_nodes = len(g.labels)
    
    # ===== Transductive 設定：創建不同時間階段的圖 =====
    print(f"\n{'='*70}")
    print("創建 Transductive 時間圖...")
    print(f"{'='*70}")
    
    # 訓練圖：所有節點 + t < t1 的邊
    g_train_temporal = create_temporal_subgraph(g, g.train_edge_mask)
    # 將對應的序列資訊加入
    g_train_temporal.lens_in = in_lens_train
    g_train_temporal.lens_out = out_lens_train
    print(f"訓練圖: {g_train_temporal.num_nodes} 節點, {g_train_temporal.edge_index.shape[1]} 邊")
    
    # 驗證圖：所有節點 + t < t2 的邊
    g_val_temporal = create_temporal_subgraph(g, g.val_edge_mask)
    g_val_temporal.lens_in = in_lens_val
    g_val_temporal.lens_out = out_lens_val
    print(f"驗證圖: {g_val_temporal.num_nodes} 節點, {g_val_temporal.edge_index.shape[1]} 邊")
    
    # 測試圖：所有節點 + 所有邊
    g_test_temporal = create_temporal_subgraph(g, g.test_edge_mask)
    g_test_temporal.lens_in = in_lens_test
    g_test_temporal.lens_out = out_lens_test
    print(f"測試圖: {g_test_temporal.num_nodes} 節點, {g_test_temporal.edge_index.shape[1]} 邊")
    print(f"{'='*70}\n")
    
    # 準備訓練資料（使用全部的 sentences，因為所有節點都可見）
    sens_selector = PreSentences_light()
    in_feats = 0
    
    # 訓練節點：在訓練時間內最後出現的節點
    train_nid = torch.nonzero(g.train_label, as_tuple=True)[0]
    
    if args.oversample and args.oversample > args.anomaly_rate:
        from imblearn.over_sampling import RandomOverSampler
        oversample = RandomOverSampler(sampling_strategy=args.oversample, random_state=args.random_state)
        nid_resampled, _ = oversample.fit_resample(train_nid.reshape(-1, 1), g.labels[g.train_label])
        train_nid = torch.as_tensor(nid_resampled.reshape(-1))
    
    val_nid = torch.nonzero(g.val_label, as_tuple=True)[0]
    test_nid = torch.nonzero(g.test_label, as_tuple=True)[0]
    
    if args.undersample:
        sampler = BalancedSampler(g.labels[train_nid])
    else:
        sampler = None
    
    # 創建 DataLoader（使用對應時間階段的圖）
    loader_train = DualNeighborSampler(g_train_temporal.edge_index,
            sizes=[25, 10],
            node_idx=train_nid,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=None if sampler else True,)
    
    loader_val = DualNeighborSampler(g_val_temporal.edge_index,
        node_idx=val_nid,
        sizes=[25, 10],
        batch_size=int(args.batch_size/2),)

    loader_test = DualNeighborSampler(g_test_temporal.edge_index,
        node_idx=test_nid,
        sizes=[25, 10],
        batch_size=int(args.batch_size/2),)

    best = []
    for repeat in range(5):
        # Training loop
        avg = 0
        iter_tput = []
        pred_pos = []
        batch_rate = []
        log_every_epoch = 5
        best_val = 0.0
        best_test = {f'Best_Precision': 0, 
                     f'Best_Recall': 0, 
                     f'Best_F1': 0, 
                     f'Best_AUC': 0.5}
        
        # 創建模型
        model = Binary_Classifier(in_feats, args.num_hidden, args.num_outputs, rnn_in_channels, 
                                    num_layers=args.num_layers, rnn_agg=args.rnn_agg, encoder_layer=args.model, 
                                    concat_feature=args.concat_feature, 
                                    dropout=args.dropout, emb_first=args.emb_first, gnn_norm=args.gnn_norm, 
                                    lstm_norm=args.lstm_norm, graph_op=args.graph_op, decoder_layers=args.decoder_layers, 
                                    aggr=args.aggr,
                                    use_cross_attention=args.use_cross_attention, 
                                    cross_attn_heads=args.cross_attn_heads)

        if args.reweight:
            weights = torch.FloatTensor([1/(g.labels[g.train_label]==0).sum().item(), 
                                        1/(g.labels[g.train_label]==1).sum().item()]).to(device)
            print('使用類別重加權')
        else:
            weights = None
        loss_fcn = nn.CrossEntropyLoss(weight=weights)
        
        weight_params = []
        for pname, p in model.encoder.named_parameters():
            if 'proj' in pname or 'adp' in pname:
                weight_params += [p]
        all_params = model.parameters()
        params_id = list(map(id, weight_params))
        other_params = list(filter(lambda p: id(p) not in params_id, all_params))
        optimizer = optim.Adam([
                {'params': other_params, 'lr': args.lr * 0.1},  # 🔧 降低學習率 10 倍
                {'params': weight_params, 'lr': args.weight_lr * 0.1}]) # 🔧 降低學習率 10 倍
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                        milestones=[10, 15], gamma=0.5)
        
        model_save_path = f'./model/best_model_{args.data}_repeat_{repeat}.pth'
        
        print(f"\n{'='*70}")
        print(f"開始第 {repeat+1}/5 次重複訓練")
        print(f"{'='*70}")
        
        for epoch in range(args.num_epochs):   
            model = model.to(device)
            model.train()
            batch_loss = 0
            tic = time.time()
            nodes_num = 0
            batch_len = len(loader_train)
            fa = []
            
            # ===== 訓練階段：使用訓練時間的序列 =====
            for loader_id, (sub_graph, subset, batch_size) in enumerate(loader_train):
                sub_graph = sub_graph.to(device)
                # 使用訓練時間過濾後的序列
                in_pack, lens_in = sens_selector.select(subset, in_sentences_train, g_train_temporal.lens_in)
                out_pack, lens_out = sens_selector.select(subset, out_sentences_train, g_train_temporal.lens_out)
                in_pack = in_pack.to(device)
                out_pack = out_pack.to(device)
                
                batch_pred, a = model(in_pack, out_pack, lens_in, lens_out, sub_graph)
                loss = loss_fcn(batch_pred[:batch_size], g.labels[subset][:batch_size].to(device))
                
                optimizer.zero_grad()
                loss.backward()
                
                    # 🔧 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                
                optimizer.step()
                scheduler.step()
                
                batch_loss += loss.item()
                nodes_num += batch_size
                fa.append(a)
                
            avg_loss = batch_loss / batch_len
            print(f"Epoch {epoch+1}/{args.num_epochs} | Loss: {avg_loss:.4f} | Nodes: {nodes_num}")
            
            # Log parameter weight
            fa = torch.cat(fa, 0)
            fa_mean = fa.mean(0).reshape(-1)
            fa_std = fa.std(0).reshape(-1)
            
            toc = time.time()
            if (epoch+1) % log_every_epoch == 0:
                # ===== 驗證和測試：分別使用對應時間的序列 =====
                val_results = evaluate_light_with_sequences(
                    model, g_val_temporal, loader_val, sens_selector, 
                    in_sentences_val, out_sentences_val, device, phase='val'
                )
                test_results = evaluate_light_with_sequences(
                    model, g_test_temporal, loader_test, sens_selector,
                    in_sentences_test, out_sentences_test, device, phase='test'
                )
                
                print(f"[驗證] ", end="")
                for item, value in val_results.items():
                    print(f"{item}:{value:.4f} ", end="")
                print()
                
                print(f"[測試] ", end="")
                for item, value in test_results.items():
                    print(f"{item}:{value:.4f} ", end="")
                print()
                
                if val_results[f'F1_val'] > best_val:
                    best_val = val_results[f'F1_val']
                    test_values = list(test_results.values())
                    for i, (item, value) in enumerate(best_test.items()):
                        best_test[item] = test_values[i]
                        
                    print(f'  → 新的最佳驗證 F1: {best_val:.4f}。儲存模型到 {model_save_path}')
                    torch.save(model.state_dict(), model_save_path)

        best.append(list(best_test.values()))
        print(f"\n重複 {repeat+1} 完成！最佳測試結果: F1={best_test['Best_F1']:.4f}, AUC={best_test['Best_AUC']:.4f}")
        print(f"{'='*70}\n")
    
    best = torch.as_tensor(best)
    best_all = {'Best_Precision_mean': best.mean(0)[0].item(), 
                'Best_Recall_mean': best.mean(0)[1].item(), 
                'Best_F1_mean': best.mean(0)[2].item(), 
                'Best_AUC_mean': best.mean(0)[3].item(),
                'Best_Precision_std': best.std(0)[0].item(), 
                'Best_Recall_std': best.std(0)[1].item(), 
                'Best_F1_std': best.std(0)[2].item(), 
                'Best_AUC_std': best.std(0)[3].item()}
    
    print(f"\n{'='*70}")
    print("最終結果（5次重複的平均與標準差）")
    print(f"{'='*70}")
    for key, value in best_all.items():
        print(f"{key}: {value:.4f}")
    print(f"{'='*70}\n")
        
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default='HI_Small')
    argparser.add_argument('--model', type=str, default='dualcata-tanh-4')
    argparser.add_argument('--data-name', type=str, default='PyG_BTC_2015')
    argparser.add_argument('--use-unlabeled', type=str, default='SEMI', help="Regard unlabeled samples as negative or not.")
    argparser.add_argument('--scale', type=str, default='minmax')
    argparser.add_argument('--rnn', type=str, default='gru', help="{lstm, gru}")
    argparser.add_argument('--rnn-feat', type=str, default='e', help="{e, nne}")
    argparser.add_argument('--rnn-agg', type=str, default='max', help="{last, max, mean, sum}")
    argparser.add_argument('--lstm-norm', type=str, default='ln', help="{ln, bn, none}")
    argparser.add_argument('--gnn-norm', type=str, default='bn')
    argparser.add_argument('--sort-by', type=str, default='a', help="Sort txes by")
    argparser.add_argument('--length', type=int, default=32, help="the maximum length of sentences")
    argparser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--emb-first', type=int, default=1, help="Whether to embeds the input of RNN first")
    argparser.add_argument('--concat-feature', type=int, default=0)
    argparser.add_argument('--train-rate', type=float, default=0.6)
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--graph-op', type=str, default=None)
    argparser.add_argument('--graph-type', type=str, default='MultiDi')
    argparser.add_argument('--feature-type', type=str, default='node')
    argparser.add_argument('--neighbor-size', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-outputs', type=int, default=2)
    argparser.add_argument('--decoder-layers', type=int, default=2)
    argparser.add_argument('--rnn_in_channels', type=int, default=8)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--random-state', type=int, default=5211)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--lr', type=float, default=0.05)
    argparser.add_argument('--weight-lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.2)
    argparser.add_argument('--anomaly_rate', type=float, default=None)
    argparser.add_argument('--reweight', type=bool, default=False)
    argparser.add_argument('--undersample', type=bool, default=False)
    argparser.add_argument('--oversample', type=float, default=None)
    argparser.add_argument('--num-workers', type=int, default=16, help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true', help="Perform the sampling process on the GPU. Must have 0 workers.")
    
    # Cross-Attention 相關參數
    argparser.add_argument('--use-cross-attention', action='store_true', help="Use cross-attention between in/out sequences")
    argparser.add_argument('--cross-attn-heads', type=int, default=2, help="Number of attention heads for cross-attention")

    argparser.set_defaults(directed=True)
    argparser.set_defaults(aggr='add')
    

    arguments = argparser.parse_args()
    main(arguments)