#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/17 3:29
# @Author : ZM7
# @File : DGSR
# @Software: PyCharm
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast  # Add this import at the top of your file
# from transformer import Transformer
import sys
import math 

# class Transformer(nn.Module):
#     '''
#         The transformer-based semantic fusion in SeHGNN.
#     '''
#     def __init__(self, n_channels, num_heads=1, att_drop=0.2, act='none'):
#         super(Transformer, self).__init__()
#         self.n_channels = n_channels
#         self.num_heads = num_heads
#         # assert self.n_channels % (self.num_heads * 4) == 0
#         assert self.n_channels % self.num_heads == 0
#         self.query = nn.Linear(self.n_channels, self.n_channels//4)
#         self.key   = nn.Linear(self.n_channels, self.n_channels//4)
#         self.value = nn.Linear(self.n_channels, self.n_channels)

#         self.gamma = nn.Parameter(torch.tensor([0.]))
#         self.att_drop = nn.Dropout(att_drop)
#         if act == 'sigmoid':
#             self.act = torch.nn.Sigmoid()
#         elif act == 'relu':
#             self.act = torch.nn.ReLU()
#         elif act == 'leaky_relu':
#             self.act = torch.nn.LeakyReLU(0.2)
#         elif act == 'none':
#             self.act = lambda x: x
#         else:
#             assert 0, f'Unrecognized activation function {act} for class Transformer'

#         self.reset_parameters()

#     def reset_parameters(self):
#         for k, v in self._modules.items():
#             if hasattr(v, 'reset_parameterxfxfxs'):
#                 v.reset_parameters()
#         nn.init.constant_(self.gamma, 0.1)  # Initialize gamma to a small non-zero value
#         nn.init.zeros_(self.gamma)

#     def forward(self, x, mask=None):
#         # print('Hi ')
#         B, M, C = x.size() # batchsize, num_metapaths, channels
#         H = self.num_heads
#         if mask is not None:
#             assert mask.size() == torch.Size((B, M))
        
#         f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
#         g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
#         h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

#         beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
#         beta = self.att_drop(beta)
#         if mask is not None:
#             beta = beta * mask.view(B, 1, 1, M)
#             beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)
#         # print('beta',beta)
#         # print('f',f)
#         # print('g',g)
#         # print('h',h)
#         # print('self.gamma',self.gamma)
#         # print('beta @ h',beta @ h)
#         o = self.gamma * (beta @ h) # [B, H, M, -1]
#         # print('o',o)

#         # print('o.permute(0,2,1,3).reshape((B, M, C)) + x',o.permute(0,2,1,3).reshape((B, M, C)))
#         return o.permute(0,2,1,3).reshape((B, M, C)) + x

class DGSR(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True):
        super(DGSR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = input_dim
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.layer_num = layer_num
        self.time = time
        self.last_item = last_item
        # long- and short-term encoder
        self.user_long = user_long
        self.item_long = item_long
        self.user_short = user_short
        self.item_short = item_short
        # update function
        self.user_update = user_update
        self.item_update = item_update
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )


        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size).to(self.device) 
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size).to(self.device) 
        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False).to(self.device) 
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False).to(self.device) 
        self.layers = nn.ModuleList([DGSRLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop,
                                                self.user_long, self.user_short, self.item_long, self.item_short,
                                                self.user_update, self.item_update,self.item_embedding) for _ in range(self.layer_num)])
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, neg_tar=None, is_training=False):
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
        unified_embedding = self.unified_map(torch.cat(user_layer, -1))
        score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0).to(self.device))
        if is_training:
            return score
        else:
            neg_embedding = self.item_embedding(neg_tar)
            # print('neg_tar',neg_tar)
            # print('neg_tar shape',neg_tar.shape)

            score_neg = torch.matmul(unified_embedding.unsqueeze(1), neg_embedding.transpose(2, 1)).squeeze(1)
            # print('score_neg',score_neg)
            # print('score_neg shape',score_neg.shape)
            # sys.exit("Terminating the program.")      

            return score, score_neg

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)



class DGSRLayers(nn.Module):
    def __init__(self, in_feats, out_feats, user_max_length, item_max_length, feat_drop=0.2, attn_drop=0.2, user_long='orgat', user_short='att',
                 item_long='orgat', item_short='att', user_update='residual', item_update='residual', item_embedding=None,K=4):
        super(DGSRLayers, self).__init__()
        self.series_table_sequential_data = pd.read_csv('/projects/academic/haimonti/atiwari4/DGSR-master/Series_table_Movielens_sample.csv')
        self.hidden_size = in_feats
        # self.transformer_fusion = Transformer(n_channels=self.hidden_size, num_heads=1, att_drop=attn_drop)

        self.item_embedding = item_embedding  # Store the item_embedding

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
            else torch.device("mps") if torch.backends.mps.is_available()
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
        # attention+ attention mechanism
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
       # update all nodes
       #print('graph update',g)
    #    if 'batch' in g.ndata:
    #        print("The graph 'g' is a batched graph.")
    #        batch_info = g.ndata['batch']  # Access batch IDs
    #        print(f"Batch IDs for all nodes: {batch_info}")
    #    else:
    #        print("The graph 'g' is NOT a batched graph.")
       subgraph_sizes = g.batch_num_nodes('item').to(self.device) 
       subgraph_ids = torch.arange(len(subgraph_sizes), device=subgraph_sizes.device).repeat_interleave(subgraph_sizes)
       g.nodes['item'].data['subgraph_id'] = subgraph_ids
       #print(f"Subgraph IDs for all nodes: {subgraph_ids}")
       # Step 1: Check the number of subgraphs
       num_subgraphs = len(subgraph_sizes)  # Count the number of subgraphs
    #    print(f"Number of subgraphs in g: {num_subgraphs}")
    #    print(f"Nodes in each subgraph: {subgraph_sizes}")
       original_ids =  g.nodes['item'].data[dgl.NID]
       local_item_ids = g.nodes('item')
    #    print(f"Local Item IDs: {local_item_ids}")
    #    print(f"Original IDs of 'item' nodes: {original_ids}")
 
       g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                           'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
      
       node_ids = g.nodes['item'].data[dgl.NID]  # Get the original IDs
       node_idx = (node_ids == 5).nonzero(as_tuple=True)[0]  # Find the matching node
  
        # Step 2: Fetch the embedding for the node
       item_embedding = g.nodes['item'].data['item_h'][node_idx]
    #    print('node_ids',node_ids)
    #    print('node_idx',node_idx)
    #    print('item_embedding',item_embedding)
   
       return g


    def item_message_func(self, edges):
        #print('edges',edges._graph)
        # print('edges.src',edges.src['user_h'])
        # print('edges.src',edges.src['user_h'].shape)
        # print('edges.dst',edges.dst['item_h'].shape)       
        dic = {}
        dic['time'] = edges.data['time']
        dic['user_h'] = edges.src['user_h']
        dic['item_h'] = edges.dst['item_h']
        return dic

    def item_reduce_func(self, nodes):
        h = []
        s=[]
        item_to_series = {row['item_id']: row for _, row in self.series_table_sequential_data.iterrows()}
        series_to_subsequent_items = {
            series_id: self.series_table_sequential_data[self.series_table_sequential_data['series_id'] == series_id]
            for series_id in self.series_table_sequential_data['series_id'].unique()
        }
        embeddings = nodes._graph.nodes['item'].data['item_h'] 
        subgraph_ids = nodes._graph.nodes['item'].data['subgraph_id'] 
        item_ids = nodes._graph.nodes['item'].data[dgl.NID] 
        s = []
        
        batch_item_ids = nodes.data[dgl.NID]
        batch_subgraph_ids = nodes.data['subgraph_id'] 
        item_info_batch = [item_to_series.get(int(item_id.item()), None) for item_id in batch_item_ids]
        subsequent_items_batch = [
            series_to_subsequent_items[item_info['series_id']]
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
            aggregated_embeddings = torch.zeros(len(subsequent_item_ids), self.hidden_size, device=self.device)
            subsequent_pos_embeddings = torch.stack(
                [torch.tensor(ast.literal_eval(pe), device=self.device) for pe in valid_subsequent_items['positional_embedding']]
            ) 

            subgraph_mask = (subgraph_ids.unsqueeze(1) == batch_subgraph_ids[i])  
            item_mask = (item_ids.unsqueeze(1) == subsequent_item_ids)  
            combined_mask = subgraph_mask & item_mask 
            valid_indices = combined_mask.any(dim=0)
            valid_embeddings = []

            for i, valid in enumerate(valid_indices):
                if valid:
                    matched_indices = torch.where(combined_mask[:, i])[0]  
                    valid_embeddings.append(embeddings[matched_indices].mean(dim=0))  
                else:
                    valid_embeddings.append(self.item_embedding(subsequent_item_ids[i]))

            valid_embeddings = torch.stack(valid_embeddings)           
            adjusted_embeddings = valid_embeddings * subsequent_pos_embeddings

            aggregated_embeddings = adjusted_embeddings.mean(dim=0) 
            s.append(aggregated_embeddings)
  
        s = [tensor.to(self.device) for tensor in s] 
        s = torch.stack(s).to(self.device)
        # h.append(s)


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

        return {'item_h': (s)} 

        # print('s',s)
        # print('self.agg_gate_i(torch.cat(h,-1))',self.agg_gate_i(torch.cat(h,-1)))
        # if len(h) == 1:
        #     return {'item_h': h[0]}
        # else:
        #     return {'item_h': self.agg_gate_i(torch.cat(h,-1))} 

    # def item_reduce_func(self, nodes):
    #     h = []
    #     s=[]

    #     item_to_series = {row['item_id']: row for _, row in self.series_table.iterrows()}
    #     embeddings = nodes._graph.nodes['item'].data['item_h']

    #     # Step 4: Handle series-based aggregation
    #     subgraph_ids = nodes._graph.nodes['item'].data['subgraph_id']  # Subgraph IDs for all nodes
    #     item_ids = nodes._graph.nodes['item'].data[dgl.NID]  # Corresponding item IDs


    #     subgraph_id = nodes.data['subgraph_id']
    #     original_ids = nodes.data[dgl.NID] 
    #     #print(f"Subgraph IDs for nodes in current NodeBatch: {subgraph_id}")
    #     #print(f"Current Node IDs (current NodeBatch) for node type '{nodes.ntype}': {original_ids}")


    #     for idx, item_id in enumerate(nodes.data[dgl.NID]):
    #         item_id = int(item_id.item())  # Convert tensor to int
    #         target_subgraph_id = nodes.data['subgraph_id'][idx].item()  # Index ensures we pick the correct subgraph ID

    #         # Debugging: Print the mappings
    #         #print(f"Item ID: {item_id}, Subgraph ID: {target_subgraph_id}")
    #         #if item_id in self.series_table['item_id'].values:
    #             # Retrieve series and position of the current item
    #         item_info = self.series_table[self.series_table['item_id'] == item_id].iloc[0]
    #         series_id = item_info['series_id']
    #         position_in_series = item_info['position_in_series']
    #         # print('series_id',series_id)
    #         # print('position_in_series',position_in_series)
    #         if series_id != 'Standalone':
    #             # Retrieve all subsequent items in the series
    #             subsequent_items = self.series_table[
    #                 (self.series_table['series_id'] == series_id) &
    #                 (self.series_table['position_in_series'] > position_in_series)
    #             ]
    #             # print('subsequent_items',subsequent_items)

    #             if subsequent_items.empty:
    #                 # If no subsequent items, append a default embedding
    #                 #print(f"Item ID {item_id} is the last in its series. Adding default embedding.")
    #                 default_embedding = torch.zeros(self.hidden_size)
    #                 s.append(default_embedding)
    #                 #print('h',item_id, 'length',len(h) )

    #             else:
    #                 aggregated_embedding = torch.zeros(self.hidden_size).to(self.device)

    #                 for _, subsequent_item in subsequent_items.iterrows():
    #                     subsequent_item_id = subsequent_item['item_id']
    #                     #print('subsequent_item_id',subsequent_item_id)

    #                     # Match subgraph ID and item ID to retrieve embedding
    #                     subgraph_mask = (subgraph_ids == target_subgraph_id)
    #                     item_mask = (item_ids == subsequent_item_id)
    #                     combined_mask = subgraph_mask & item_mask

    #                     # Retrieve embedding
    #                     if combined_mask.any():  # If the item exists in the same subgraph
    #                         #filtered_embeddings = nodes._graph.nodes['item'].data['item_h'][combined_mask]
    #                         filtered_embeddings = embeddings[combined_mask]

    #                     else:
    #                         filtered_embeddings = self.item_embedding(torch.tensor(subsequent_item_id, dtype=torch.long,device=self.device))

    #                     positional_embedding = torch.tensor(ast.literal_eval(subsequent_item['positional_embedding'])).to(self.device)

    #                     adjusted_embedding = filtered_embeddings * positional_embedding
    #                     aggregated_embedding += adjusted_embedding.squeeze(0).to(self.device)  # Remove the extra dimension from adjusted_embedding if present
    #             # Append the aggregated embedding for all subsequent items
    #                 s.append(aggregated_embedding)
    #                 #print('h',item_id, 'length',len(h) )
    #         else:
    #                 # Handle standalone items (assign default embeddings)
    #                 default_embedding = torch.zeros(self.hidden_size)  # Replace with learnable param if needed
    #                 s.append(default_embedding)
    #                 #print('h',item_id, 'length',len(h) )
    #     # print('s',s)
    #     s = [tensor.to(self.device) for tensor in s] 
    #     s = torch.stack(s).to(self.device)
    #     # print('s',s) 
    #     h.append(s)
    #     # print('length',len(h))
    #     order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
    #     re_order = nodes.mailbox['time'].shape[1] -order -1
    #     length = nodes.mailbox['item_h'].shape[0]
    #     if self.item_long == 'orgat':
    #         e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
    #                /torch.sqrt(torch.tensor(self.hidden_size).float())
    #         alpha = self.atten_drop(F.softmax(e_ij, dim=1))
    #         if len(alpha.shape) == 2:
    #             alpha = alpha.unsqueeze(2)
    #         h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
    #         h.append(h_long)
    #         # print('h_long',h_long.shape)
    #     elif self.item_long == 'gru':
    #         rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
    #         _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
    #         h.append(hidden_u.squeeze(0))
    #         #print('hidden_u.squeeze(0)',hidden_u.squeeze(0))
    #     last = torch.argmax(nodes.mailbox['time'], 1)
    #     last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
    #     if self.item_short == 'att':
    #         e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
    #             torch.tensor(self.hidden_size).float())
    #         alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
    #         if len(alpha1.shape) == 2:
    #             alpha1 = alpha1.unsqueeze(2)
    #         h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
    #         h.append(h_short)
    #         # print('h_short',h_short)

    #     elif self.item_short == 'last':
    #         h.append(last_em.squeeze())
    #     if len(h) == 1:
    #         return {'item_h': h[0]}
    #     else:
    #         return {'item_h': self.agg_gate_i(torch.cat(h,-1))} 


    # def item_reduce_func(self, nodes):
    #     h = []

    #     # # Precompute mappings for efficient lookups
    #     # item_to_series = {row['item_id']: row for _, row in self.series_table.iterrows()}

    #     # # Batch processing for series-based aggregation
    #     # subgraph_ids = nodes._graph.nodes['item'].data['subgraph_id']
    #     # item_ids = nodes._graph.nodes['item'].data[dgl.NID]
    #     # subgraph_id = nodes.data['subgraph_id']
    #     # original_ids = nodes.data[dgl.NID]

    #     # # Retrieve embeddings for all items in batch
    #     # embeddings = nodes._graph.nodes['item'].data['item_h']

    #     # # Handle series information
    #     # for idx, item_id in enumerate(nodes.data[dgl.NID].tolist()):
    #     #     item_id = int(item_id)
    #     #     target_subgraph_id = subgraph_id[idx].item()

    #     #     if item_id in item_to_series:
    #     #         item_info = item_to_series[item_id]
    #     #         series_id = item_info['series_id']
    #     #         position_in_series = item_info['position_in_series']

    #     #         if series_id != 'Standalone':
    #     #             subsequent_items = self.series_table[
    #     #                 (self.series_table['series_id'] == series_id) &
    #     #                 (self.series_table['position_in_series'] > position_in_series)
    #     #             ]

    #     #             if subsequent_items.empty:
    #     #                 default_embedding = torch.zeros(self.hidden_size, device=self.device)
    #     #                 h.append(default_embedding)
    #     #             else:
    #     #                 aggregated_embedding = torch.zeros(self.hidden_size, device=self.device)
    #     #                 for _, subsequent_item in subsequent_items.iterrows():
    #     #                     subsequent_item_id = subsequent_item['item_id']
    #     #                     subgraph_mask = (subgraph_ids == target_subgraph_id)
    #     #                     item_mask = (item_ids == subsequent_item_id)
    #     #                     combined_mask = subgraph_mask & item_mask

    #     #                     if combined_mask.any():
    #     #                         filtered_embeddings = embeddings[combined_mask]
    #     #                     else:
    #     #                         filtered_embeddings = self.item_embedding(
    #     #                             torch.tensor(subsequent_item_id, dtype=torch.long, device=self.device)
    #     #                         )

    #     #                     positional_embedding = torch.tensor(
    #     #                         ast.literal_eval(subsequent_item['positional_embedding']), device=self.device
    #     #                     )
    #     #                     adjusted_embedding = filtered_embeddings * positional_embedding
    #     #                     aggregated_embedding += adjusted_embedding.squeeze(0)

    #     #                 h.append(aggregated_embedding)
    #     #         else:
    #     #             default_embedding = torch.zeros(self.hidden_size, device=self.device)
    #     #             h.append(default_embedding)

    #     order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
    #     re_order = nodes.mailbox['time'].shape[1] -order -1
    #     length = nodes.mailbox['item_h'].shape[0]
    #     if self.item_long == 'orgat':
    #         e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
    #                /torch.sqrt(torch.tensor(self.hidden_size).float())
    #         alpha = self.atten_drop(F.softmax(e_ij, dim=1))
    #         if len(alpha.shape) == 2:
    #             alpha = alpha.unsqueeze(2)
    #         h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
    #         h.append(h_long)
    #         #print('h_long',h_long)
    #     elif self.item_long == 'gru':
    #         rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
    #         _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
    #         h.append(hidden_u.squeeze(0))
    #         #print('hidden_u.squeeze(0)',hidden_u.squeeze(0))
    #     last = torch.argmax(nodes.mailbox['time'], 1)
    #     last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
    #     if self.item_short == 'att':
    #         e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / torch.sqrt(
    #             torch.tensor(self.hidden_size).float())
    #         alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
    #         if len(alpha1.shape) == 2:
    #             alpha1 = alpha1.unsqueeze(2)
    #         h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
    #         h.append(h_short)
    #         #print('h_short',h_short)

    #     elif self.item_short == 'last':
    #         h.append(last_em.squeeze())
    #     # Combine series-based and short/long-term information
    #     #h = torch.stack(h).to(self.device)
    #     # if len(h) > 1:
    #     #     # Ensure all tensors in h have the same dimensions
    #     #     h = [tensor if tensor.ndim == 2 else tensor.unsqueeze(0) for tensor in h]

    #     #     # Determine the maximum size for padding
    #     #     max_size = max(tensor.size(0) for tensor in h)

    #     #     # Pad tensors to the maximum size
    #     #     h_padded = [
    #     #         F.pad(tensor, (0, 0, 0, max_size - tensor.size(0))) if tensor.size(0) < max_size else tensor
    #     #         for tensor in h
    #     #     ]

    #     #     # Ensure all tensors are on the correct device
    #     #     h_padded = [tensor.to(self.device) for tensor in h_padded]

    #     #     # Concatenate along the last dimension
    #     #     concatenated_h = torch.cat(h_padded, dim=-1)

    #     #     # Ensure agg_gate_i matches concatenated_h dimensions
    #     #     input_dim = concatenated_h.shape[-1]
    #     #     if not hasattr(self, "agg_gate_i") or self.agg_gate_i.in_features != input_dim:
    #     #         self.agg_gate_i = nn.Linear(input_dim, self.hidden_size, bias=False).to(self.device)

    #     #     # Apply the linear layer
    #     #     item_h = self.agg_gate_i(concatenated_h)
    #     # else:
    #     #     item_h = h[0]  # Single tensor, no concatenation needed
    #     if len(h) == 1:
    #         return {'item_h': h[0]}
    #     else:
    #         return {'item_h': self.agg_gate_i(torch.cat(h,-1))}

    #     # return {'item_h': item_h}


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
        else torch.device("mps") if torch.backends.mps.is_available()
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


def neg_generate(user, data_neg, neg_num=100):
    neg = np.zeros((len(user), neg_num), np.int32)
    for i, u in enumerate(user):
        neg[i] = np.random.choice(data_neg[u.item()], neg_num, replace=False)
        # print('neg[i]',neg[i])
    # sys.exit("Terminating the program.")           
    return neg


def collate_test(data, user_neg):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
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
        print('Hi I am in collate')
    return torch.tensor(user).long(), dgl.batch(graph), torch.tensor(label).long(), torch.tensor(last_item).long(), torch.Tensor(neg_generate(user, user_neg)).long()




