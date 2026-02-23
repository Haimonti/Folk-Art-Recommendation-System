import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast  # Add this import at the top of your file
import sys
import math 

class HSAL(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True):
        super(HSAL, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update = user_update
        self.item_update = item_update
        #self.device = torch.device("cuda")
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            #else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )


        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size).to(self.device) 
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size).to(self.device) 
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False).to(self.device) 
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False).to(self.device) 
        self.layers = nn.ModuleList([HSALLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop,
                                                self.user_long, self.user_short, self.item_long, self.item_short,
                                                self.user_update, self.item_update,self.item_embedding) for _ in range(self.layer_num)])
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, pos_tar=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        g = g.to(self.device)

        g.nodes['user'].data['user_h'] = self.user_embedding(g.nodes['user'].data['user_id'].to(self.device))
        g.nodes['item'].data['item_h'] = self.item_embedding(g.nodes['item'].data['item_id'].to(self.device))
        
        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)
                user_layer.append(graph_user(g, user_index, feat_dict['user'].to(self.device)))
            if self.last_item:
                item_embed = graph_item(g, last_item_index, feat_dict['item'].to(self.device))
                user_layer.append(item_embed)
        
        # This is the user's final representation
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))

        if is_training:
            # --- BPR CHANGE START ---
            # Instead of scoring ALL items, we only score the Positive and Negative pairs
            
            # 1. Get embeddings for the specific positive and negative items
            pos_embedding = self.item_embedding(pos_tar)
            neg_embedding = self.item_embedding(neg_tar)

            # 2. Calculate scores (Dot Product)
            # unified_embedding: [batch_size, hidden_size]
            # pos/neg_embedding: [batch_size, hidden_size]
            pos_score = torch.sum(unified_embedding * pos_embedding, dim=1)
            neg_score = torch.sum(unified_embedding * neg_embedding, dim=1)
            
            return pos_score, neg_score
            # --- BPR CHANGE END ---
        else:
            # Evaluation remains mostly the same (ranking against all items or specific negatives)
            # Note: For efficiency in validation, you might want to keep the full matrix multiply
            score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0).to(self.device))
            
            # If you need specific negative scores for testing metrics:
            if neg_tar is not None:
                neg_embedding = self.item_embedding(neg_tar)
                score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
                return score, score_neg
            
            return score, None

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)



class HSALLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, user_long='orgat', user_short='att',
                 item_long='orgat', item_short='att', user_update='residual', item_update='residual', item_embedding=None, K=4):
        super(HSALLayers, self).__init__()
        
        # --- OPTIMIZATION START ---
        # 1. Load Data
        csv_file = 'Series_table_sinusodial_Series_table_MovieLens_all.csv' # <--- UPDATE THIS IF NEEDED FOR MOVIELENS_ALL
        print(f"Loading Series Table from {csv_file}...")
        self.series_table_sequential_data = pd.read_csv(csv_file)
        
        # 2. Pre-parse 'positional_embedding' from String to Tensor ONCE
        print("Pre-parsing positional embeddings (removing ast.literal_eval from loop)...")
        # We use a helper to safely parse and convert
        def parse_embedding(x):
            try:
                # Convert string list to tensor
                return torch.tensor(ast.literal_eval(x), dtype=torch.float)
            except:
                # Fallback for empty/errors
                return torch.zeros(in_feats)

        self.series_table_sequential_data['positional_embedding'] = self.series_table_sequential_data['positional_embedding'].apply(parse_embedding)
        
        # 3. Build Dictionaries using the pre-parsed data
        print("Building series lookup maps...")
        self.item_to_series = {
            row['item_id']: row 
            for _, row in self.series_table_sequential_data.iterrows()
        }
        
        # Store subsequent items as a lookup dict
        self.series_to_subsequent_items = {
            series_id: self.series_table_sequential_data[self.series_table_sequential_data['series_id'] == series_id]
            for series_id in self.series_table_sequential_data['series_id'].unique()
        }
        print("HSAL Layer Initialization Complete.")
        # --- OPTIMIZATION END ---

        self.hidden_size = in_feats
        self.item_embedding = item_embedding

        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        self.user_update_m = user_update
        self.item_update_m = item_update
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.K = torch.tensor(K)       
        if self.user_long in ['orgat', 'gcn', 'gru'] and self.user_short in ['last','att', 'att1']:
            self.agg_gate_u = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.item_long in ['orgat', 'gcn', 'gru'] and self.item_short in ['last', 'att', 'att1']:
            self.agg_gate_i = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        if self.user_long in ['gru']:
            self.gru_u = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.item_long in ['gru']:
            self.gru_i = nn.GRU(input_size=in_feats, hidden_size=in_feats, batch_first=True)
        if self.user_update_m == 'norm':
            self.norm_user = nn.LayerNorm(self.hidden_size)
        if self.item_update_m == 'norm':
            self.norm_item = nn.LayerNorm(self.hidden_size)
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)
        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.user_update_m in ['concat', 'rnn']:
            self.user_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        if self.item_update_m in ['concat', 'rnn']:
            self.item_update = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        
        if self.user_short in ['last', 'att']:
            self.last_weight_u = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.item_short in ['last', 'att']:
            self.last_weight_i = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.item_long in ['orgat']:
            self.i_time_encoding = nn.Embedding(self.user_max_length, self.hidden_size)
            self.i_time_encoding_k = nn.Embedding(self.user_max_length, self.hidden_size)
        if self.user_long in ['orgat']:
            self.u_time_encoding = nn.Embedding(self.item_max_length, self.hidden_size)
            self.u_time_encoding_k = nn.Embedding(self.item_max_length, self.hidden_size)


    def user_update_function(self, user_now, user_old):
        if self.user_update_m == 'residual':
            return F.elu(user_now + user_old)
        elif self.user_update_m == 'gate_update':
            pass
        elif self.user_update_m == 'concat':
            return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'light':
            pass
        elif self.user_update_m == 'norm':
            return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn':
            return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        else:
            print('error: no user_update')
            exit()

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual':
            return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat':
            return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'light':
            pass
        elif self.item_update_m == 'norm':
            return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn':
            return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        else:
            print('error: no item_update')
            exit()

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).to(self.device)
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).to(self.device)
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user']
            item_ = feat_dict['item']
            if self.user_long in ['gcn']:
                g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).to(self.device)
            if self.item_long in ['gcn']:
                g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).to(self.device)
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        f_dict = {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}
        return f_dict

    def graph_update(self, g):
       subgraph_sizes = g.batch_num_nodes('item').to(self.device) 
       subgraph_ids = torch.arange(len(subgraph_sizes), device=subgraph_sizes.device).repeat_interleave(subgraph_sizes)
       g.nodes['item'].data['subgraph_id'] = subgraph_ids

       num_subgraphs = len(subgraph_sizes)

       original_ids =  g.nodes['item'].data[dgl.NID]
       local_item_ids = g.nodes('item')
 
       g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                           'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
      
       node_ids = g.nodes['item'].data[dgl.NID]
       node_idx = (node_ids == 5).nonzero(as_tuple=True)[0]
  
       item_embedding = g.nodes['item'].data['item_h'][node_idx]
       return g


    def item_message_func(self, edges):     
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    def item_reduce_func(self, nodes):
        h = []
        s = []
        
        # === PRE-COMPUTED LOOKUP (NO DATAFRAME CREATION) ===
        embeddings = nodes._graph.nodes['item'].data['item_h'] 
        subgraph_ids = nodes._graph.nodes['item'].data['subgraph_id'] 
        item_ids = nodes._graph.nodes['item'].data[dgl.NID] 
        
        batch_item_ids = nodes.data[dgl.NID]
        batch_subgraph_ids = nodes.data['subgraph_id'] 

        # Lookup item info using pre-computed dict
        item_info_batch = [self.item_to_series.get(int(item_id.item()), None) for item_id in batch_item_ids]
        
        # Lookup subsequent items using pre-computed dict
        subsequent_items_batch = [
            self.series_to_subsequent_items[item_info['series_id']]
            if item_info is not None and item_info['series_id'] != 'Standalone' else None
            for item_info in item_info_batch
        ]                           
        
        valid_items = [
            subsequent_items[subsequent_items['position_in_series'] > item_info['position_in_series']]
            if subsequent_items is not None and item_info is not None else None
            for subsequent_items, item_info in zip(subsequent_items_batch, item_info_batch)
        ]
        
        for i, valid_subsequent_items in enumerate(valid_items):
            if valid_subsequent_items is None or valid_subsequent_items.empty:
                s.append(torch.zeros(self.hidden_size, device=self.device))  
                continue

            subsequent_item_ids = torch.tensor(valid_subsequent_items['item_id'].values, device=self.device, dtype=torch.long)
            
            # === FAST TENSOR STACK (NO AST.LITERAL_EVAL) ===
            # We simply stack the pre-parsed tensors from the dataframe column
            subsequent_pos_embeddings = torch.stack(valid_subsequent_items['positional_embedding'].tolist()).to(self.device)
            # ===============================================

            subgraph_mask = (subgraph_ids.unsqueeze(1) == batch_subgraph_ids[i])  
            item_mask = (item_ids.unsqueeze(1) == subsequent_item_ids)  
            combined_mask = subgraph_mask & item_mask 
            valid_indices = combined_mask.any(dim=0)
            valid_embeddings = []

            for k, valid in enumerate(valid_indices):
                if valid:
                    matched_indices = torch.where(combined_mask[:, k])[0]  
                    valid_embeddings.append(embeddings[matched_indices].mean(dim=0))  
                else:
                    valid_embeddings.append(self.item_embedding(subsequent_item_ids[k]))

            valid_embeddings = torch.stack(valid_embeddings)           
            adjusted_embeddings = valid_embeddings * subsequent_pos_embeddings

            aggregated_embeddings = adjusted_embeddings.mean(dim=0) 
            s.append(aggregated_embeddings)

        s = [tensor.to(self.device) for tensor in s] 
        s = torch.stack(s).to(self.device)

        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] -order -1
        length = nodes.mailbox['item_h'].shape[0]
        if self.item_long == 'orgat':
            e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
                   /torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
                torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)

        elif self.item_short == 'last':
            h.append(last_em.squeeze())

        return {'item_h': (self.agg_gate_i(torch.cat(h,-1))+s)} 


    def user_message_func(self, edges):
        dic = {}
        dic['time'] = edges.data['time']
        dic['item_h'] = edges.src['item_h']
        dic['user_h'] = edges.dst['user_h']
        return dic

    def user_reduce_func(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1),1)
        re_order = nodes.mailbox['time'].shape[1] - order -1
        length = nodes.mailbox['user_h'].shape[0]
        if self.user_long == 'orgat':
            e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) *nodes.mailbox['user_h'],
                             dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2:
                alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['item_h'], dim=2)/torch.sqrt(torch.tensor(self.hidden_size).float())
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2:
                alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())

        if len(h) == 1:
            return {'user_h': h[0]}
        else:
            return {'user_h': self.agg_gate_u(torch.cat(h,-1))}

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1).to(user_index.device)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]

def order_update(edges):
    dic = {}
    dic['order'] = torch.sort(edges.data['time'])[1]
    dic['re_order'] = len(edges.data['time']) - dic['order']
    return dic


def collate(data):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        #else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    user = []
    user_l = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['user'])
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0]) 
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        #print('batch graph',da[0][0])
        #break               
    return torch.tensor(user_l).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long()

def collate_bpr(data, user_neg_data):
    user_l = []
    graph = []
    label = []
    last_item = []
    neg_item = [] # List for the negative samples

    for da in data:
        # Extract the user ID (ensure it's an integer for lookup)
        # da[1]['user'] is a tensor like tensor([5]), so we use .item()
        u_id = da[1]['user'].item()
        
        user_l.append(da[1]['u_alis'])
        graph.append(da[0][0]) 
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        
        # --- BPR Negative Sampling ---
        # 1. Retrieve the list of valid negatives for this user
        valid_negatives = user_neg_data[u_id]
        
        # 2. Randomly sample ONE item from that list
        # We use np.random.choice for efficiency
        neg_id = np.random.choice(valid_negatives)
        
        neg_item.append(neg_id)
        # -----------------------------

    return (torch.tensor(user_l).long(), 
            dgl.batch(graph), 
            torch.tensor(label).long(), 
            torch.tensor(last_item).long(), 
            torch.tensor(neg_item).long()) # Returns the specific negative items

def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u.item()], neg_num, replace=False)
        # print('neg[i]',neg[i])
    # sys.exit("Terminating the program.")           
    return neg


def collate_test(data, user_neg):
    #device = torch.device('cuda')
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        #else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    user = []
    graph = []
    label = []
    last_item = []
    for da in data:
        user.append(da[1]['u_alis'])
        graph.append(da[0][0])

        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])
        #print('Hi I am in collate')
    return torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.Tensor(neg_generate(user, user_neg)).long()




