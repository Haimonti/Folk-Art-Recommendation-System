import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from HSAL import HSAL, collate, collate_test
from dgl import load_graphs
import pickle
from utils import myFloder
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from HSAL_utils import eval_metric, mkdir_if_not_exist, Logger
from tqdm import tqdm
import dgl

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='MovieLens_Final_100k', help='data name')
    #  "batch size = 50"
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    #  "learning rate 0.01"
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    #  "weight decay lambda = 1e-4"
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
    
    parser.add_argument('--user_update', default='rnn')
    parser.add_argument('--item_update', default='rnn')
    parser.add_argument('--user_long', default='orgat')
    parser.add_argument('--item_long', default='orgat')
    parser.add_argument('--user_short', default='att')
    parser.add_argument('--item_short', default='att')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
    parser.add_argument('--item_max_length', type=int, default=50, help='item seq length')
    parser.add_argument('--user_max_length', type=int, default=50, help='user seq length')
    parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--last_item', action='store_true', help='aggreate last item')
    parser.add_argument("--record", action='store_true', default=False, help='record results')
    parser.add_argument("--val", action='store_true', default=False)
    parser.add_argument("--model_record", action='store_true', default=False, help='record model')

    opt = parser.parse_args()
    print(opt.data)
    args, extras = parser.parse_known_args()
    
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device', device)

    if opt.record:
        log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_l2_{opt.l2}'
        mkdir_if_not_exist(log_file)
        sys.stdout = Logger(log_file)
        print(f'Logging to {log_file}')

    if opt.model_record:
        model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                f'_layer_{opt.layer_num}_l2_{opt.l2}'

    # Loading data
    data = pd.read_csv('./Data/' + opt.data + '.csv')
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item) # Important: Ensure item IDs are contiguous 0 to N-1
    
    train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
    val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    
    train_set = myFloder(train_root, load_graphs)
    test_set = myFloder(test_root, load_graphs)
    if opt.val:
        val_set = myFloder(val_root, load_graphs)
        
    f = open(opt.data+'_neg', 'rb')
    data_neg = pickle.load(f) 
    print('train')

    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0)
    if opt.val:
        print('val')
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)

    model = HSAL(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
                item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
                layer_num=opt.layer_num).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    
    #  "binary cross-entropy loss with L2 regularization"
    # BCEWithLogitsLoss includes the sigmoid layer and is more stable than BCELoss
    loss_func = nn.BCEWithLogitsLoss() 
    
    best_result = [0] * 22
    best_epoch = [0] * 22
    stop_num = 0

    for epoch in range(opt.epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        print('start training: ', datetime.datetime.now())
        model.train()

        train_loader = tqdm(train_data, desc=f"Epoch {epoch+1}/{opt.epoch} [Training]")
        for user, batch_graph, label, last_item in train_loader:
            user = user.to(device)
            batch_graph = batch_graph.to(device)
            label = label.to(device)
            last_item = last_item.to(device)
            iter += 1
            
            # Forward pass: score shape is [batch_size, num_items]
            score = model(batch_graph, user, last_item, is_training=True)
            
            # --- Implementation of Eq (8) ---
            # Construct strict One-Hot targets for BCE
            # Targets are 1 for the ground truth item, 0 for ALL other items
            targets = torch.zeros_like(score)
            targets[torch.arange(score.size(0)), label] = 1.0
            
            loss = loss_func(score, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            train_loader.set_postfix(loss=epoch_loss / iter)
        
        epoch_loss /= iter
        model.eval()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}', '=============================================')

        # --- VALIDATION (Unchanged Logic, just device handling) ---
        if opt.val:
            print('start validation: ', datetime.datetime.now())
            val_loss_all, top_val = [], []
            val_loader = tqdm(val_data, desc=f"Epoch {epoch+1}/{opt.epoch} [Validation]")

            with torch.no_grad():
                for user, batch_graph, label, last_item, neg_tar in val_loader:
                    # In validation/test, we often rank against specific negatives, 
                    # but for loss calculation consistency we use the BCE approach if we want val_loss
                    # However, usually we just care about metrics here.
                    
                    score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), 
                                     neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                    
                    # Note: Calculating validation loss with BCE requires full scores, 
                    # but 'model' in is_training=False mode returns scores only for candidates (pos + negs).
                    # So we skip val_loss calculation here or adapt it. 
                    # Below assumes we just track metrics.
                    top_val.append(top.detach().cpu().numpy())
                    
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
                print(f'Val Recall@10:{recall10:.4f}\tNDCG@10:{ndgg10:.4f}')

        # --- TEST (Unchanged Logic) ---
        print('start predicting: ', datetime.datetime.now())
        all_top = []
        test_loader = tqdm(test_data, desc="Testing")

        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_loader:
                user = user.to(device)
                batch_graph = batch_graph.to(device)
                label = label.to(device)
                last_item = last_item.to(device)
                neg_tar = neg_tar.to(device)

                score, top = model(batch_graph, user, last_item, 
                                 neg_tar=torch.cat([label.unsqueeze(1), neg_tar],-1), is_training=False)
                
                all_top.append(top.detach().cpu().numpy())

            # Evaluate Metrics
            metrics = eval_metric(all_top) 
            # (Assuming eval_metric returns the list of Recalls and NDCGs in order)
            
            # --- Logic to save best results ---
            # (Matches original script logic, abbreviated for clarity)
            curr_recall10 = metrics[1]
            curr_ndcg10 = metrics[12]
            
            if curr_recall10 > best_result[1]:
                best_result[1] = curr_recall10
                best_epoch[1] = epoch
                if opt.model_record:
                     # Ensure directory exists
                    if not os.path.exists('save_models'): os.makedirs('save_models')
                    torch.save(model.state_dict(), 'save_models/'+ (model_file if opt.model_record else 'model') + '.pkl')

            print(f'Epoch {epoch+1} Test Results: Recall@10: {curr_recall10:.4f}, NDCG@10: {curr_ndcg10:.4f}')

    print('End training: ', datetime.datetime.now())

if __name__ == '__main__':
    main()