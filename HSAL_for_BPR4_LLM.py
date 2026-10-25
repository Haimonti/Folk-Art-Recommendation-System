import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
import math

class HSAL(nn.Module):
    def __init__(self, user_num, item_num, input_dim, item_max_length, user_max_length, feat_drop=0.2, attn_drop=0.2,
                 user_long='orgat', user_short='att', item_long='ogat', item_short='att', user_update='rnn',
                 item_update='rnn', last_item=True, layer_num=3, time=True,
                 series_file=None, relation_file=None, relation_mode='none',
                 relation_cap=10, relation_confidence_threshold=0.5,
                 relation_weight=1.0, relation_dropout=0.0, add_inverse_relations=True):
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size).to(self.device) 
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size).to(self.device) 

        # --- PRE-COMPUTE GLOBALLY ONCE TO AVOID CPU RAM OOM (KILLED) ---
        # Note: If you parameterized this filename, replace the hardcoded string with your opt.data variable!
        csv_file = series_file or 'Series_table_sinusodial_Goodreads_fantasy_paranormal_HSAL_100k.csv'
        print(f"Loading and optimizing Series Table from {csv_file}...")
        
        # 1. Use pure Python dicts instead of Pandas DataFrames to save CPU RAM
        series_df = pd.read_csv(csv_file)
        item_to_series = {}
        series_to_items = {}
        
        # 2. Extract values cleanly
        for _, row in series_df.iterrows():
            i_id = int(row['item_id'])
            s_id = row['series_id']
            pos = int(row['position_in_series'])
            
            # Parse the embedding safely
            emb_str = row['positional_embedding']
            if isinstance(emb_str, str):
                emb = torch.tensor(ast.literal_eval(emb_str), dtype=torch.float)
            elif isinstance(emb_str, list):
                emb = torch.tensor(emb_str, dtype=torch.float)
            else:
                emb = torch.zeros(input_dim)
                
            item_to_series[i_id] = {'series_id': s_id, 'position': pos}
            
            if s_id != 'Standalone':
                if s_id not in series_to_items:
                    series_to_items[s_id] = []
                series_to_items[s_id].append((pos, i_id, emb))
                
        # 3. Sort series items by position to ensure strict chronological sequence
        for s_id in series_to_items:
            series_to_items[s_id].sort(key=lambda x: x[0])
            
        subsequent_ids_list = []
        subsequent_pos_list = []
        MAX_SUB_CAP = 50  # Cap the maximum sequence to prevent GPU VRAM explosions
        
        for item_id in range(self.item_num):
            info = item_to_series.get(item_id, None)
            if info is not None and info['series_id'] != 'Standalone':
                s_id = info['series_id']
                curr_pos = info['position']
                
                # Get all items in the same series with a strictly greater position
                subs = [x for x in series_to_items[s_id] if x[0] > curr_pos]
                if len(subs) > 0:
                    ids = [x[1] for x in subs][:MAX_SUB_CAP]
                    pos_embs = torch.stack([x[2] for x in subs][:MAX_SUB_CAP])
                    
                    subsequent_ids_list.append(ids)
                    subsequent_pos_list.append(pos_embs)
                    continue
                    
            subsequent_ids_list.append([])
            subsequent_pos_list.append(torch.zeros((0, input_dim)))
            
        max_subsequent = max(max([len(l) for l in subsequent_ids_list]), 1)
        
        self.subsequent_ids_table = torch.zeros((self.item_num, max_subsequent), dtype=torch.long)
        self.subsequent_pos_emb_table = torch.zeros((self.item_num, max_subsequent, input_dim), dtype=torch.float)
        self.subsequent_mask = torch.zeros((self.item_num, max_subsequent), dtype=torch.bool)
        
        for item_id in range(self.item_num):
            ids = subsequent_ids_list[item_id]
            if len(ids) > 0:
                self.subsequent_ids_table[item_id, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                self.subsequent_pos_emb_table[item_id, :len(ids)] = subsequent_pos_list[item_id]
                self.subsequent_mask[item_id, :len(ids)] = True
                
        self.subsequent_ids_table = self.subsequent_ids_table.to(self.device)
        self.subsequent_pos_emb_table = self.subsequent_pos_emb_table.to(self.device)
        self.subsequent_mask = self.subsequent_mask.to(self.device)
        print(f"Pre-computation complete. Max sequence length safely capped at {max_subsequent}.")

        # LLM relation enhancement. Expected columns:
        # source_item_id,target_item_id,relation,confidence
        self.relation_mode = relation_mode.lower()
        relation_ids = [[] for _ in range(self.item_num)]
        relation_types = [[] for _ in range(self.item_num)]
        relation_scores = [[] for _ in range(self.item_num)]
        enabled = {x.strip().lower() for x in self.relation_mode.split(',')}
        if 'both' in enabled or 'all' in enabled:
            enabled.update({'prequel', 'sequel'})
        if self.relation_mode != 'none' and relation_file:
            rel_df = pd.read_csv(relation_file)
            aliases = {
                'source_item_id': ['source_item_id', 'source_id', 'item_id', 'src'],
                'target_item_id': ['target_item_id', 'target_id', 'related_item_id', 'dst'],
                'relation': ['relation', 'relation_type', 'type'],
                'confidence': ['confidence', 'score', 'similarity']
            }
            resolved = {}
            for canonical, choices in aliases.items():
                resolved[canonical] = next((c for c in choices if c in rel_df.columns), None)
            required = ['source_item_id', 'target_item_id', 'relation']
            missing = [c for c in required if resolved[c] is None]
            if missing:
                raise ValueError(f"Relation file is missing columns for: {missing}")
            seen = set()
            rows = []
            for _, row in rel_df.iterrows():
                src_id = int(row[resolved['source_item_id']])
                dst_id = int(row[resolved['target_item_id']])
                rel = str(row[resolved['relation']]).strip().lower()
                conf = float(row[resolved['confidence']]) if resolved['confidence'] else 1.0
                if rel not in {'prequel', 'sequel'} or rel not in enabled:
                    continue
                if not (0 <= src_id < self.item_num and 0 <= dst_id < self.item_num):
                    continue
                if src_id == dst_id or conf < relation_confidence_threshold:
                    continue
                rows.append((src_id, dst_id, rel, conf))
                if add_inverse_relations:
                    inv = 'sequel' if rel == 'prequel' else 'prequel'
                    if inv in enabled:
                        rows.append((dst_id, src_id, inv, conf))
            rows.sort(key=lambda x: x[3], reverse=True)
            for src_id, dst_id, rel, conf in rows:
                key = (src_id, dst_id, rel)
                if key in seen or len(relation_ids[src_id]) >= relation_cap:
                    continue
                seen.add(key)
                relation_ids[src_id].append(dst_id)
                relation_types[src_id].append(1 if rel == 'prequel' else 2)
                relation_scores[src_id].append(conf)

        max_rel = max(max((len(x) for x in relation_ids), default=0), 1)
        self.relation_ids_table = torch.zeros((self.item_num, max_rel), dtype=torch.long, device=self.device)
        self.relation_type_table = torch.zeros((self.item_num, max_rel), dtype=torch.long, device=self.device)
        self.relation_score_table = torch.zeros((self.item_num, max_rel), dtype=torch.float, device=self.device)
        self.relation_mask = torch.zeros((self.item_num, max_rel), dtype=torch.bool, device=self.device)
        for item_id in range(self.item_num):
            n = len(relation_ids[item_id])
            if n:
                self.relation_ids_table[item_id, :n] = torch.tensor(relation_ids[item_id], device=self.device)
                self.relation_type_table[item_id, :n] = torch.tensor(relation_types[item_id], device=self.device)
                self.relation_score_table[item_id, :n] = torch.tensor(relation_scores[item_id], device=self.device)
                self.relation_mask[item_id, :n] = True
        edge_count = int(self.relation_mask.sum().item())
        print(f"LLM relation table ready: {edge_count} directed edges, cap={relation_cap}.")
        # ---------------------------------------------------------------

        if self.last_item:
            self.unified_map = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size, bias=False).to(self.device) 
        else:
            self.unified_map = nn.Linear(self.layer_num * self.hidden_size, self.hidden_size, bias=False).to(self.device) 
        
        self.layers = nn.ModuleList([
            HSALLayers(self.hidden_size, self.hidden_size, self.user_max_length, self.item_max_length, feat_drop, attn_drop,
                       self.user_long, self.user_short, self.item_long, self.item_short,
                       self.user_update, self.item_update, self.item_embedding,
                       self.subsequent_ids_table, self.subsequent_pos_emb_table, self.subsequent_mask, self.item_num,
                       relation_ids_table=self.relation_ids_table,
                       relation_type_table=self.relation_type_table,
                       relation_score_table=self.relation_score_table,
                       relation_mask=self.relation_mask,
                       relation_weight=relation_weight,
                       relation_dropout=relation_dropout) 
            for _ in range(self.layer_num)
        ])
        self.reset_parameters()

    def forward(self, g, user_index=None, last_item_index=None, pos_tar=None, neg_tar=None, is_training=False):
        feat_dict = None
        user_layer = []
        if g.device != self.device:
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

        if is_training:
            pos_embedding = self.item_embedding(pos_tar)
            neg_embedding = self.item_embedding(neg_tar)
            pos_score = torch.sum(unified_embedding * pos_embedding, dim=1)
            neg_score = torch.sum(unified_embedding * neg_embedding, dim=1)
            return pos_score, neg_score
        else:
            score = torch.matmul(unified_embedding, self.item_embedding.weight.transpose(1, 0).to(self.device))
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
                 item_long='orgat', item_short='att', user_update='residual', item_update='residual', item_embedding=None,
                 subsequent_ids_table=None, subsequent_pos_emb_table=None, subsequent_mask=None, item_num=0, K=4,
                 relation_ids_table=None, relation_type_table=None, relation_score_table=None,
                 relation_mask=None, relation_weight=1.0, relation_dropout=0.0):
        super(HSALLayers, self).__init__()
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = torch.tensor(K)       

        # --- Link Shared Lookups ---
        self.subsequent_ids_table = subsequent_ids_table
        self.subsequent_pos_emb_table = subsequent_pos_emb_table
        self.subsequent_mask = subsequent_mask
        self.item_num = item_num
        self.relation_ids_table = relation_ids_table
        self.relation_type_table = relation_type_table
        self.relation_score_table = relation_score_table
        self.relation_mask = relation_mask
        self.relation_weight = relation_weight
        self.relation_dropout = nn.Dropout(relation_dropout)
        self.prequel_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sequel_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.prequel_gate = nn.Parameter(torch.tensor(0.0))
        self.sequel_gate = nn.Parameter(torch.tensor(0.0))

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
        if self.user_update_m == 'residual': return F.elu(user_now + user_old)
        elif self.user_update_m == 'concat': return F.elu(self.user_update(torch.cat([user_now, user_old], -1)))
        elif self.user_update_m == 'norm': return self.feat_drop(self.norm_user(user_now)) + user_old
        elif self.user_update_m == 'rnn': return F.tanh(self.user_update(torch.cat([user_now, user_old], -1)))
        return user_now

    def item_update_function(self, item_now, item_old):
        if self.item_update_m == 'residual': return F.elu(item_now + item_old)
        elif self.item_update_m == 'concat': return F.elu(self.item_update(torch.cat([item_now, item_old], -1)))
        elif self.item_update_m == 'norm': return self.feat_drop(self.norm_item(item_now)) + item_old
        elif self.item_update_m == 'rnn': return F.tanh(self.item_update(torch.cat([item_now, item_old], -1)))
        return item_now

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            if self.user_long in ['gcn']: g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).to(self.device)
            if self.item_long in ['gcn']: g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).to(self.device)
            user_ = g.nodes['user'].data['user_h']
            item_ = g.nodes['item'].data['item_h']
        else:
            user_ = feat_dict['user']
            item_ = feat_dict['item']
            if self.user_long in ['gcn']: g.nodes['user'].data['norm'] = g['by'].in_degrees().unsqueeze(1).to(self.device)
            if self.item_long in ['gcn']: g.nodes['item'].data['norm'] = g['by'].out_degrees().unsqueeze(1).to(self.device)
            
        g.nodes['user'].data['user_h'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_h'] = self.item_weight(self.feat_drop(item_))
        g = self.graph_update(g)
        g.nodes['user'].data['user_h'] = self.user_update_function(g.nodes['user'].data['user_h'], user_)
        g.nodes['item'].data['item_h'] = self.item_update_function(g.nodes['item'].data['item_h'], item_)
        return {'user': g.nodes['user'].data['user_h'], 'item': g.nodes['item'].data['item_h']}

    def graph_update(self, g):
        subgraph_sizes = g.batch_num_nodes('item').to(self.device) 
        subgraph_ids = torch.arange(len(subgraph_sizes), device=subgraph_sizes.device).repeat_interleave(subgraph_sizes)
        g.nodes['item'].data['subgraph_id'] = subgraph_ids

        embeddings = g.nodes['item'].data['item_h'] 
        item_ids = g.nodes['item'].data[dgl.NID] 
        
        curr_subs_ids = self.subsequent_ids_table[item_ids]       
        curr_subs_pos = self.subsequent_pos_emb_table[item_ids]   
        curr_subs_mask = self.subsequent_mask[item_ids]           
        
        # --- OOM-SAFE O(1) BATCH LOOKUP TRICK ---
        num_subgraphs = len(subgraph_sizes)
        batch_lookup = torch.full((num_subgraphs, self.item_num), -1, device=self.device, dtype=torch.long)
        node_indices = torch.arange(len(item_ids), device=self.device)
        batch_lookup[subgraph_ids, item_ids] = node_indices
        
        found_indices = batch_lookup[subgraph_ids.unsqueeze(1), curr_subs_ids]
        valid_found_mask = found_indices >= 0
        
        safe_indices = torch.clamp(found_indices, min=0)
        matched_emb = embeddings[safe_indices]
        
        raw_embeddings = self.item_embedding(curr_subs_ids)
        valid_embeddings = torch.where(valid_found_mask.unsqueeze(-1), matched_emb, raw_embeddings)
        adjusted_embeddings = valid_embeddings * curr_subs_pos 
        
        mask_expanded = curr_subs_mask.unsqueeze(-1).float()
        sum_adjusted = (adjusted_embeddings * mask_expanded).sum(dim=1)
        subs_counts = mask_expanded.sum(dim=1)
        
        s = sum_adjusted / torch.clamp(subs_counts, min=1.0)
        s = torch.where(subs_counts > 0, s, torch.zeros_like(s))

        # Relation-aware, confidence-weighted LLM neighbor aggregation.
        relation_message = torch.zeros_like(embeddings)
        if self.relation_mask is not None:
            rel_ids = self.relation_ids_table[item_ids]
            rel_types = self.relation_type_table[item_ids]
            rel_scores = self.relation_score_table[item_ids]
            rel_mask = self.relation_mask[item_ids]
            rel_found = batch_lookup[subgraph_ids.unsqueeze(1), rel_ids]
            rel_safe = torch.clamp(rel_found, min=0)
            rel_local_emb = embeddings[rel_safe]
            rel_global_emb = self.item_embedding(rel_ids)
            rel_emb = torch.where((rel_found >= 0).unsqueeze(-1), rel_local_emb, rel_global_emb)
            prequel_emb = self.prequel_transform(rel_emb)
            sequel_emb = self.sequel_transform(rel_emb)
            typed_emb = torch.where((rel_types == 1).unsqueeze(-1), prequel_emb, sequel_emb)
            gates = torch.where(rel_types == 1, torch.sigmoid(self.prequel_gate), torch.sigmoid(self.sequel_gate))
            weights = rel_scores * gates * rel_mask.float()
            weighted = self.relation_dropout(typed_emb) * weights.unsqueeze(-1)
            denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            relation_message = weighted.sum(dim=1) / denom
            relation_message = torch.where(rel_mask.any(dim=1, keepdim=True), relation_message, torch.zeros_like(relation_message))
        # ----------------------------------------

        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
      
        g.nodes['item'].data['item_h'] = (g.nodes['item'].data['item_h'] + s
                                                + self.relation_weight * relation_message)
        return g

    def item_message_func(self, edges):     
        return {'time': edges.data['time'], 'user_h': edges.src['user_h'], 'item_h': edges.dst['item_h']}

    def item_reduce_func(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['item_h'].shape[0]
        
        if self.item_long == 'orgat':
            e_ij = torch.sum((self.i_time_encoding(re_order) + nodes.mailbox['user_h']) * nodes.mailbox['item_h'], dim=2)\
                   / math.sqrt(self.hidden_size)
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2: alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['user_h'] + self.i_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.item_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_u = self.gru_i(nodes.mailbox['user_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_u.squeeze(0))
            
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['user_h'][torch.arange(length), last, :].unsqueeze(1)
        
        if self.item_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / math.sqrt(self.hidden_size)
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2: alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['user_h'], dim=1)
            h.append(h_short)
        elif self.item_short == 'last':
            h.append(last_em.squeeze())

        return {'item_h': self.agg_gate_i(torch.cat(h, -1))} 

    def user_message_func(self, edges):
        return {'time': edges.data['time'], 'item_h': edges.src['item_h'], 'user_h': edges.dst['user_h']}

    def user_reduce_func(self, nodes):
        h = []
        order = torch.argsort(torch.argsort(nodes.mailbox['time'], 1), 1)
        re_order = nodes.mailbox['time'].shape[1] - order - 1
        length = nodes.mailbox['user_h'].shape[0]
        
        if self.user_long == 'orgat':
            e_ij = torch.sum((self.u_time_encoding(re_order) + nodes.mailbox['item_h']) * nodes.mailbox['user_h'], dim=2) \
                   / math.sqrt(self.hidden_size)
            alpha = self.atten_drop(F.softmax(e_ij, dim=1))
            if len(alpha.shape) == 2: alpha = alpha.unsqueeze(2)
            h_long = torch.sum(alpha * (nodes.mailbox['item_h'] + self.u_time_encoding_k(re_order)), dim=1)
            h.append(h_long)
        elif self.user_long == 'gru':
            rnn_order = torch.sort(nodes.mailbox['time'], 1)[1]
            _, hidden_i = self.gru_u(nodes.mailbox['item_h'][torch.arange(length).unsqueeze(1), rnn_order])
            h.append(hidden_i.squeeze(0))
            
        last = torch.argmax(nodes.mailbox['time'], 1)
        last_em = nodes.mailbox['item_h'][torch.arange(length), last, :].unsqueeze(1)
        
        if self.user_short == 'att':
            e_ij1 = torch.sum(last_em * nodes.mailbox['user_h'], dim=2) / math.sqrt(self.hidden_size)
            alpha1 = self.atten_drop(F.softmax(e_ij1, dim=1))
            if len(alpha1.shape) == 2: alpha1 = alpha1.unsqueeze(2)
            h_short = torch.sum(alpha1 * nodes.mailbox['item_h'], dim=1)
            h.append(h_short)
        elif self.user_short == 'last':
            h.append(last_em.squeeze())

        return {'user_h': h[0] if len(h) == 1 else self.agg_gate_u(torch.cat(h, -1))}

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
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    user_alis = []
    global_users = [] # Track the actual user IDs
    graph = []
    label = []
    last_item = []
    for da in data:
        user_alis.append(da[1]['u_alis'])
        global_users.append(da[1]['user']) # Grab the global ID
        graph.append(da[0][0])
        label.append(da[1]['target'])
        last_item.append(da[1]['last_alis'])

    # Pass global_users to neg_generate, but return user_alis for the graph
    return (torch.tensor(user_alis).long(), 
            dgl.batch(graph), 
            torch.tensor(label).long(), 
            torch.tensor(last_item).long(), 
            torch.Tensor(neg_generate(global_users, user_neg)).long())