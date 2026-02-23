import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
import sys

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return data

def extract_sequel_edges(data):
    if 'item_series_id' not in data.columns:
        return pd.DataFrame(columns=['src', 'dst', 'series', 'rel_pos'])

    unique_items = (
        data[['item_id', 'item_series_id']]
        .drop_duplicates()
        .dropna(subset=['item_series_id'])
    )

    edges = []

    for s, group in unique_items.groupby('item_series_id'):
        group = group.sort_values('item_id')
        items = group['item_id'].values

        for idx in range(len(items) - 1):
            edges.append({
                'src': items[idx],
                'dst': items[idx + 1],
                'series': s,
                'rel_pos': 1  # immediate sequel
            })

    return pd.DataFrame(edges)


def generate_graph(data):
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    data = data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)
    
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    
    # Extract sequel relationships dynamically from the dataframe
    sequel_df = extract_sequel_edges(data)
    
    # Base user-item edges
    graph_data = {
        ('item', 'by', 'user'): (torch.tensor(item), torch.tensor(user)),
        ('user', 'pby', 'item'): (torch.tensor(user), torch.tensor(item))
    }
    
    # Add sequel edges if they exist 
    if not sequel_df.empty:
        src_sequel = torch.tensor(sequel_df['item_id'].values)
        dst_sequel = torch.tensor(sequel_df['sequel_id'].values)
        graph_data[('item', 'sequel', 'item')] = (src_sequel, dst_sequel)
        print(f"Added {len(sequel_df)} sequel edges to the graph.")
    
    graph = dgl.heterograph(graph_data)
    print('#####################')
    print(graph.edges['sequel'].data.keys())
    print(graph.edges['by'].data.keys())
    print(graph.edges['pby'].data.keys())
    print('#####################')

    if not sequel_df.empty:
        graph.edges['sequel'].data['series'] = torch.LongTensor(
            sequel_df['series'].values
        )
        graph.edges['sequel'].data['rel_pos'] = torch.LongTensor(
            sequel_df['rel_pos'].values
        )

    # Assign time to user-item edges
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    # user -> item (pby): position of item in user's sequence
    graph.edges['pby'].data['p_ui'] = torch.LongTensor(
        data['order'].values
    )

    # item -> user (by): position of user in item's interaction sequence
    graph.edges['by'].data['p_iu'] = torch.LongTensor(
        data['u_order'].values
    )

    
    # Assign IDs
    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    
    # Ensure all items (including those potentially only in sequel chains) are mapped
    all_items = np.unique(np.concatenate([item, 
                                          sequel_df['item_id'].values if not sequel_df.empty else [], 
                                          sequel_df['sequel_id'].values if not sequel_df.empty else []]))
    graph.nodes['item'].data['item_id'] = torch.LongTensor(all_items)
    
    return graph

def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop=3, val_path=None):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    
    # Check if sequel edges exist in the graph schema
    has_sequels = 'sequel' in graph.etypes

    if len(u_seq) < 3:
        return 0, 0
    else:
        for j, t in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]
            
            # Select edges within the time window
            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
            
            edge_dict = {'by': sub_u_eid, 'pby': sub_i_eid}
            
            # [cite: 135] "If an interacted item i belongs to a sequel series... these sequel items are also included"
            if has_sequels:
                # Include all sequel edges as they are static structure
                sub_s_eid = graph.edges(form='eid', etype='sequel')
                edge_dict['sequel'] = sub_s_eid
            
            sub_graph = dgl.edge_subgraph(graph, edges=edge_dict, relabel_nodes=False)#, preserve_nodes=True)
            
            u_temp = torch.tensor([user])
            his_user = torch.tensor([user])
            
            # Step 1: Get items interacted with by the user
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
            i_temp = torch.unique(graph_i.edges(etype='by')[0])
            his_item = torch.unique(graph_i.edges(etype='by')[0])
            
            edge_i = [graph_i.edges['by'].data[dgl.NID]]
            edge_u = []
            edge_s = [] 
            
            for _ in range(k_hop-1):
                # 1. EXPAND ITEMS VIA SEQUELS [cite: 135]
                # Fix: Check for sequel neighbors before fetching users
                if has_sequels:
                    graph_s = sample_neighbors(sub_graph, {'item': i_temp}, -1, edge_dir='out', etype='sequel')
                    
                    if graph_s.num_edges('sequel') > 0:
                        edge_s.append(graph_s.edges['sequel'].data[dgl.NID])
                        sequel_items = torch.unique(graph_s.edges(etype='sequel')[1])
                        # Update candidate items to include sequels
                        i_temp = torch.unique(torch.cat([i_temp, sequel_items]))
                        his_item = torch.unique(torch.cat([his_item, sequel_items]))

                # 2. FIND USERS
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
                
                # 3. FIND ITEMS
                graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
                
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                
                edge_i.append(graph_i.edges['by'].data[dgl.NID])
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])

            all_edge_u = torch.unique(torch.cat(edge_u)) if edge_u else torch.tensor([], dtype=torch.long)
            all_edge_i = torch.unique(torch.cat(edge_i)) if edge_i else torch.tensor([], dtype=torch.long)
            
            final_edges = {'by': all_edge_i, 'pby': all_edge_u}
            
            if has_sequels and edge_s:
                final_edges['sequel'] = torch.unique(torch.cat(edge_s))
            elif has_sequels:
                final_edges['sequel'] = torch.tensor([], dtype=torch.long)

            fin_graph = dgl.edge_subgraph(sub_graph, edges=final_edges)
            
            target = u_seq[j+1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id'] == user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id'] == last_item)[0]
       
            if j < split_point-1:
                save_graphs(train_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis, 'last_alis': last_alis})
                train_num += 1
            if j == split_point - 1 - 1:
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis})
            if j == split_point - 1:
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis, 'last_alis': last_alis})
                test_num += 1
        return train_num, test_num


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3):
    user = data['user_id'].unique()
    a = Parallel(n_jobs=job)(delayed(lambda u: generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop, val_path))(u) for u in user)
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='MovieLens_Final_100k', help='data name: mixed_data')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    opt = parser.parse_args()
    
    data_path = './Data/' + opt.data + '.csv'
    graph_path = './Data/' + opt.data + '_graph.bin'
    
    # LOAD DATA WITH ITEM_SERIES_ID
    # Ensure item_series_id is read correctly (handling NaNs if necessary)
    data = pd.read_csv(data_path)
    
    # Preprocessing
    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data['time'] = data['time'].astype('int64')
    
    if not os.path.exists(graph_path):
        # Pass the full data (with item_series_id) to generate_graph
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
        
    train_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    val_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    test_path = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    users = data['user_id'].unique()
    for u in users:
        os.makedirs(train_path + str(u), exist_ok=True)
        os.makedirs(val_path + str(u), exist_ok=True)
        os.makedirs(test_path + str(u), exist_ok=True)

    print('start:', datetime.datetime.now())
    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path, job=opt.job, k_hop=opt.k_hop)
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)
    print('end:', datetime.datetime.now())