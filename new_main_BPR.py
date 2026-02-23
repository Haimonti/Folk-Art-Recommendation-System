import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
#from DGSR_o import DGSR, collate, collate_test
from HSAL_for_BPR1 import HSAL, collate, collate_test, collate_bpr
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
#from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger
from HSAL_utils import eval_metric, mkdir_if_not_exist, Logger
from tqdm import tqdm
from functools import partial
import dgl


warnings.filterwarnings('ignore')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='MovieLens_Final_100k', help='data name: mixed_data')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
    parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
    parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
    parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--last_item', action='store_true', help='aggreate last item')
    parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
    parser.add_argument("--val", action='store_true', default=False)
    parser.add_argument("--model_record", action='store_true', default=False, help='record model')

    opt = parser.parse_args()
    print(opt.data)
    args, extras = parser.parse_known_args()
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        #else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print('device',device)


    if opt.record:
        log_file = f'results/{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                f'_layer_{opt.layer_num}_l2_{opt.l2}'
        mkdir_if_not_exist(log_file)
        sys.stdout = Logger(log_file)
        print(f'Logging to {log_file}')
    if opt.model_record:
        model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_' \
                f'US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}' \
                f'_layer_{opt.layer_num}_l2_{opt.l2}'

    # loading data
    data = pd.read_csv('./Data/' + opt.data + '.csv')
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
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

    #########################################################################################
    # Create a partial function that fixes 'user_neg_data' to your loaded 'data_neg'
    train_collate_fn = partial(collate_bpr, user_neg_data=data_neg)

    train_data = DataLoader(dataset=train_set, 
                            batch_size=opt.batchSize, 
                            collate_fn=train_collate_fn, # Use the partial function
                            shuffle=True, 
                            pin_memory=True, 
                            num_workers=12)
    
    #########################################################################################
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0)
    if opt.val:
        print('val')
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)

    model = HSAL(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
                user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
                item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
                layer_num=opt.layer_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    
    #########################################################################################
    # BPR Loss: -ln(sigmoid(pos - neg))
    def bpr_loss(pos_scores, neg_scores):
        return -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    
    val_loss_func = nn.CrossEntropyLoss()

    #########################################################################################
    #loss_func = nn.CrossEntropyLoss()
    #loss_func = nn.BCEWithLogitsLoss() 
    best_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # hit5,hit10,hit20,mrr5,mrr10,mrr20
    best_epoch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    stop_num = 0
    for epoch in range(opt.epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        print('start training: ', datetime.datetime.now())
        model.train()
        print(len(train_data))
        print(type(train_data))


        train_loader = tqdm(train_data, desc=f"Epoch {epoch+1}/{opt.epoch} [Training]")
        #########################################################################################
        # Note: Added 'neg_item' to the unpacking
        for user, batch_graph, label, last_item, neg_item in train_loader:
            user = user.to(device)
            batch_graph = batch_graph.to(device)
            label = label.to(device)      # This is the Positive Item
            last_item = last_item.to(device)
            neg_item = neg_item.to(device) # This is the Negative Item (from collate_bpr)
            
            iter += 1
            
            # Pass explicit pos_tar and neg_tar to the model
            # Ensure your HSAL model forward method accepts these arguments!
            pos_score, neg_score = model(batch_graph, user, last_item, 
                                         pos_tar=label, 
                                         neg_tar=neg_item, 
                                         is_training=True)
            
            # Calculate BPR Loss
            loss = bpr_loss(pos_score, neg_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #########################################################################################

            train_loader.set_postfix(loss=epoch_loss / iter)
        
        epoch_loss /= iter
        model.eval()
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}', '=============================================')

        # val
        if opt.val:
            print('start validation: ', datetime.datetime.now())
            val_loss_all, top_val = [], []
            val_loader = tqdm(val_data, desc=f"Epoch {epoch+1}/{opt.epoch} [Validation]")

            with torch.no_grad:
                for user, batch_graph, label, last_item, neg_tar in val_loader:
                    score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                    val_loss = val_loss_func(score, label.to(device))
                    val_loss_all.append(val_loss.append(val_loss.item()))
                    top_val.append(top.detach().cpu().numpy())
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
                print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f'
                    '\tNDGG10@10:%.4f\tNDGG@20:%.4f' %
                    (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))

        # test
        print('start predicting: ', datetime.datetime.now())
        all_top, all_label, all_length = [], [], []
        iter = 0
        all_loss = []
        test_loader = tqdm(test_data, desc="Testing")

        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_loader:
                iter+=1
                user = user.to(device)
                batch_graph = batch_graph.to(device)
                label = label.to(device)
                last_item = last_item.to(device)
                neg_tar = neg_tar.to(device)

                score, top = model(batch_graph, user, last_item, neg_tar=torch.cat([label.unsqueeze(1), neg_tar],-1),  is_training=False)
                test_loss = val_loss_func(score, label)
                all_loss.append(test_loss.item())
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.cpu().numpy())
                test_loader.set_postfix(test_loss=np.mean(all_loss))
            recall5, recall10, recall20, recall30, recall40, recall50, recall60, recall70, recall80, recall90, recall100,  ndgg5, ndgg10, ndgg20, ndgg30, ndgg40, ndgg50, ndgg60, ndgg70, ndgg80, ndgg90, ndgg100 = eval_metric(all_top)
            if recall5 > best_result[0]:
                best_result[0] = recall5
                best_epoch[0] = epoch
                stop = False
            if recall10 > best_result[1]:
                if opt.model_record:
                    torch.save(model.state_dict(), 'save_models/'+ model_file + '.pkl')
                best_result[1] = recall10
                best_epoch[1] = epoch
                stop = False
            if recall20 > best_result[2]:
                best_result[2] = recall20
                best_epoch[2] = epoch
                stop = False
            if recall30 > best_result[3]:
                best_result[3] = recall30
                best_epoch[3] = epoch
                stop = False  
            if recall40 > best_result[4]:
                best_result[4] = recall40
                best_epoch[4] = epoch
                stop = False      
            if recall50 > best_result[5]:
                best_result[5] = recall50
                best_epoch[5] = epoch
                stop = False
            if recall60 > best_result[6]:
                best_result[6] = recall60
                best_epoch[6] = epoch
                stop = False  
            if recall70 > best_result[7]:
                best_result[7] = recall70
                best_epoch[7] = epoch
                stop = False 
            if recall80 > best_result[8]:
                best_result[8] = recall80
                best_epoch[8] = epoch      
            if recall90 > best_result[9]:
                best_result[9] = recall90
                best_epoch[9] = epoch  
            if recall100 > best_result[10]:
                best_result[10] = recall100
                best_epoch[10] = epoch                                     
            if ndgg5 > best_result[11]:
                best_result[11] = ndgg5
                best_epoch[11] = epoch
                stop = False
            if ndgg10 > best_result[12]:
                best_result[12] = ndgg10
                best_epoch[12] = epoch
                stop = False
            if ndgg20 > best_result[13]:
                best_result[13] = ndgg20
                best_epoch[13] = epoch
                stop = False
            if ndgg30 > best_result[14]:
                best_result[14] = ndgg30
                best_epoch[14] = epoch
                stop = False
            if ndgg40 > best_result[15]:
                best_result[15] = ndgg40
                best_epoch[15] = epoch
                stop = False            
            if ndgg50 > best_result[16]:
                best_result[16] = ndgg50
                best_epoch[16] = epoch
                stop = False 
            if ndgg60 > best_result[17]:
                best_result[17] = ndgg60
                best_epoch[17] = epoch
                stop = False  
            if ndgg70 > best_result[18]:
                best_result[18] = ndgg70
                best_epoch[18] = epoch
                stop = False   
            if ndgg80 > best_result[19]:
                best_result[19] = ndgg80
                best_epoch[19] = epoch
                stop = False    
            if ndgg90 > best_result[20]:
                best_result[20] = ndgg90
                best_epoch[20] = epoch    
                stop = False    
            if ndgg100 > best_result[21]:
                best_result[21] = ndgg100
                best_epoch[21] = epoch    
                stop = False                                                                        
            if stop:
                stop_num += 1
            else:
                stop_num = 0

            print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tRecall@30:%.4f\tRecall@40:%.4f\tRecall@50:%.4f\tRecall@60:%.4f\tRecall@70:%.4f\tRecall@80:%.4f\tRecall@90:%.4f\tRecall@100:%.4f\tNDGG@5:%.4f'
                '\tNDGG10@10:%.4f\tNDGG@20:%.4f\tNDGG@30:%.4f\tNDGG@40:%.4f\tNDGG@50:%.4f\tNDGG@60:%.4f\tNDGG@70:%.4f\tNDGG@80:%.4f\tNDGG@90:%.4f\tNDGG@100:%.4f\tEpoch:%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d' %
                (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3],
                best_result[4], best_result[5], best_result[6], best_result[7], best_result[8], best_result[9], best_result[10],
                best_result[11], best_result[12], best_result[13], best_result[14], best_result[15], best_result[16],  best_result[17],  best_result[18],  best_result[19],  best_result[20],  best_result[21], best_epoch[0], best_epoch[1],
                best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5], best_epoch[6], best_epoch[7], best_epoch[8],
                best_epoch[9], best_epoch[10], best_epoch[11], best_epoch[12], best_epoch[13], best_epoch[14], best_epoch[15], best_epoch[16], best_epoch[17], best_epoch[18], best_epoch[19], best_epoch[20], best_epoch[21]
                ))
            

    print('End training: ', datetime.datetime.now())

if __name__ == '__main__':
    main()