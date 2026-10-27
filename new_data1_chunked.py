#!/usr/bin/env python
# Chunked version of new_data1.py.
#
# new_data1.py reads one CSV, renumbers IDs from it, builds the full user-book graph and cuts one
# subgraph per training example. This version does the same from N user-disjoint chunk CSVs:
#
#   add-chunk    read one chunk, extend the global user/book ID maps, and add its edges to the
#                saved graph (graph after chunk 1, then after chunk 2, ...)
#   build-graph  build the full graph in one shot from every chunk already added (faster at scale)
#   finalize     after the last chunk: renumber users/books in original-file order (same numbers as
#                a single-file run) and relabel the graph
#   samples      cut the per-example subgraphs for the users of selected chunks; run one job per
#                chunk in parallel, all writing to the same Newdata/ folder
#
# Samples must only be cut after every chunk is in the graph: a user's subgraph pulls in other
# users who read the same books, and those users can live in any chunk.
#
# State lives in Data/<name>_chunked/: users.csv, books.csv (raw id -> global id) and one
# chunk_XX.npz of edges per chunk. The graph is saved to Data/<name>_graph, the same path
# new_data1.py uses, and samples go to the same Newdata/ layout, so training code is unchanged.

import argparse
import datetime
import glob
import os
from multiprocessing import Pool

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import save_graphs
from dgl.sampling import select_topk


def check_name(name):
    """Chunked outputs must never reuse an original dataset name (they would overwrite Data/<name>_graph)."""
    if not name.endswith('_chunked'):
        raise SystemExit(f"--data '{name}' must end with '_chunked' so original outputs are never overwritten")


def require_all_chunks(name, n_chunks):
    """Graph-wide steps (samples, series table, negatives) need every chunk: a user's subgraph and a
    book's sequels can come from any chunk."""
    done = len(added_chunks(name))
    if done != n_chunks:
        raise SystemExit(f'{done} of {n_chunks} chunks added to {name}; add all chunks first')
    if not is_finalized(name):
        raise SystemExit(f'{name} is not finalized: run "new_data1_chunked.py finalize" after adding all chunks')


def state_dir(name):
    return f'Data/{name}_chunked'


def graph_path(name):
    return f'Data/{name}_graph'


def load_id_map(path, with_first_row=False):
    if not os.path.exists(path):
        return ({}, {}) if with_first_row else {}
    df = pd.read_csv(path, dtype={'raw_id': str})
    id_map = dict(zip(df['raw_id'], df['id']))
    if with_first_row:
        return id_map, dict(zip(df['raw_id'], df['first_row']))
    return id_map


def save_id_map(id_map, first_row, path):
    raw = list(id_map.keys())
    pd.DataFrame({'raw_id': raw, 'id': [id_map[r] for r in raw],
                  'first_row': [first_row[r] for r in raw]}).to_csv(path, index=False)


def is_finalized(name):
    return os.path.exists(os.path.join(state_dir(name), 'finalized'))


def added_chunks(name):
    manifest = os.path.join(state_dir(name), 'chunks.txt')
    if not os.path.exists(manifest):
        return []
    with open(manifest) as f:
        return [line.strip() for line in f if line.strip()]


def refine_time(user, time):
    """Same result as new_data1.refine_time applied per user, for all users at once.

    Rows are sorted by (user, time) keeping file order for ties (mergesort in the original);
    a time that is not strictly after the previous one in the same user's sequence is pushed
    forward by an increasing gap, exactly as the original loop does.
    """
    order = np.lexsort((time, user))
    user, time = user[order], time[order].copy()
    gap = 1
    for i in range(len(time) - 1):
        if user[i] != user[i + 1]:
            gap = 1
            continue
        if time[i] >= time[i + 1]:
            time[i + 1] = time[i + 1] + gap
            gap += 1
    return order, time


def add_chunk(name, chunk_csv, update_graph=True):
    os.makedirs(state_dir(name), exist_ok=True)
    chunk_key = os.path.abspath(chunk_csv)
    done = added_chunks(name)
    if chunk_key in done:
        print(f'{chunk_csv} already added, skipping')
        return
    if is_finalized(name):
        raise SystemExit(f'{name} is already finalized; start a new --data name to add more chunks')
    users, user_first = load_id_map(os.path.join(state_dir(name), 'users.csv'), with_first_row=True)
    books, book_first = load_id_map(os.path.join(state_dir(name), 'books.csv'), with_first_row=True)

    data = pd.read_csv(chunk_csv, dtype={'user_id': str, 'item_id': str})
    chunk_no = len(done) + 1
    # position of each row in the original single file (written by split_hsal_csv_by_user.py);
    # finalize uses it to give users/books the same numbers a single-file run would.
    # Without it, rows count in chunk order.
    if 'orig_row' in data.columns:
        rows = data['orig_row'].values.astype('int64')
    else:
        rows = chunk_no * 10**12 + np.arange(len(data), dtype='int64')
    seen_users = set(data['user_id'].unique()) & users.keys()
    if seen_users:
        raise ValueError(f'{len(seen_users)} users in {chunk_csv} already appear in an earlier chunk; '
                         f'chunks must be user-disjoint')

    # users: new ids continue after the previous chunk, in order of first appearance
    first_seen = pd.unique(data['user_id'].values)
    for raw in first_seen:
        users[raw] = len(users)
    # books are shared across chunks: only books not seen before get a new id
    for raw in pd.unique(data['item_id'].values):
        if raw not in books:
            books[raw] = len(books)
    for raw, row in pd.Series(rows).groupby(data['user_id'].values).min().items():
        user_first[raw] = row
    for raw, row in pd.Series(rows).groupby(data['item_id'].values).min().items():
        book_first[raw] = min(row, book_first.get(raw, row))

    user = data['user_id'].map(users).values.astype('int64')
    item = data['item_id'].map(books).values.astype('int64')
    order, time = refine_time(user, data['time'].values.astype('int64'))
    user, item = user[order], item[order]

    np.savez(os.path.join(state_dir(name), f'chunk_{chunk_no:02d}.npz'), user=user, item=item, time=time)
    save_id_map(users, user_first, os.path.join(state_dir(name), 'users.csv'))
    save_id_map(books, book_first, os.path.join(state_dir(name), 'books.csv'))
    with open(os.path.join(state_dir(name), 'chunks.txt'), 'a') as f:
        f.write(chunk_key + '\n')
    print(f'chunk {chunk_no}: {len(user):,} edges, {len(first_seen):,} new users, '
          f'totals {len(users):,} users / {len(books):,} books')

    if update_graph:
        extend_graph(name, user, item, time, len(users), len(books))


def set_node_ids(graph):
    graph.nodes['user'].data['user_id'] = torch.arange(graph.num_nodes('user'), dtype=torch.long)
    graph.nodes['item'].data['item_id'] = torch.arange(graph.num_nodes('item'), dtype=torch.long)
    return graph


def extend_graph(name, user, item, time, n_users, n_books):
    """Load the saved graph (if any), append this chunk's edges and save it again."""
    path = graph_path(name)
    user_t, item_t, time_t = torch.tensor(user), torch.tensor(item), torch.LongTensor(time)
    if os.path.exists(path):
        graph = dgl.load_graphs(path)[0][0]
        # nodes for new users / books first, so new edges never point past the node range
        graph.add_nodes(n_users - graph.num_nodes('user'), ntype='user')
        graph.add_nodes(n_books - graph.num_nodes('item'), ntype='item')
        graph.add_edges(item_t, user_t, data={'time': time_t}, etype='by')
        graph.add_edges(user_t, item_t, data={'time': time_t}, etype='pby')
    else:
        graph = dgl.heterograph({('item', 'by', 'user'): (item_t, user_t),
                                 ('user', 'pby', 'item'): (user_t, item_t)},
                                num_nodes_dict={'user': n_users, 'item': n_books})
        graph.edges['by'].data['time'] = time_t
        graph.edges['pby'].data['time'] = time_t
    graph = set_node_ids(graph)
    save_graphs(path, graph)
    print(f'graph: {graph.num_nodes("user"):,} users, {graph.num_nodes("item"):,} books, '
          f'{graph.num_edges("by"):,} edges -> {path}')


def load_chunk_edges(name, chunk_numbers=None):
    files = sorted(glob.glob(os.path.join(state_dir(name), 'chunk_*.npz')))
    if chunk_numbers:
        files = [f for f in files if int(os.path.basename(f)[6:8]) in chunk_numbers]
    parts = [np.load(f) for f in files]
    return (np.concatenate([p['user'] for p in parts]),
            np.concatenate([p['item'] for p in parts]),
            np.concatenate([p['time'] for p in parts]))


def build_graph(name):
    """Build the full graph once from every added chunk (same result as many add-chunk steps)."""
    user, item, time = load_chunk_edges(name)
    # new_data1.generate_graph leaves edges grouped by book, then by time; keep that order
    order = np.lexsort((user, time, item))
    user, item, time = user[order], item[order], time[order]
    n_users = len(load_id_map(os.path.join(state_dir(name), 'users.csv')))
    n_books = len(load_id_map(os.path.join(state_dir(name), 'books.csv')))
    graph = dgl.heterograph({('item', 'by', 'user'): (torch.tensor(item), torch.tensor(user)),
                             ('user', 'pby', 'item'): (torch.tensor(user), torch.tensor(item))},
                            num_nodes_dict={'user': n_users, 'item': n_books})
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    graph = set_node_ids(graph)
    save_graphs(graph_path(name), graph)
    print(f'graph: {n_users:,} users, {n_books:,} books, {len(user):,} edges -> {graph_path(name)}')


def finalize(name):
    """Renumber users and books in the order they first appear in the original file.

    add-chunk numbers users chunk by chunk, so user numbers differ from a single-file run.
    The ID numbers matter: when more than user_max_length neighbour users are found,
    generate_user keeps the ones with the largest numbers. Renumbering in original-file order
    (what pd.factorize does in new_data1.py) makes the chunked graph and samples identical to a
    single-file run. The graph built chunk by chunk is relabelled, not rebuilt from scratch.
    """
    if is_finalized(name):
        print(f'{name} already finalized')
        return
    # anything built before finalize uses the old numbering and would no longer match the graph
    stale = [p for p in glob.glob(f'Newdata/{name}_*') + [f'{name}_neg', f'Series_table_sinusodial_Series_table_{name}.csv']
             if os.path.exists(p)]
    if stale:
        raise SystemExit(f'outputs built before finalize would use the old ID numbers; delete them first: {stale}')
    users, user_first = load_id_map(os.path.join(state_dir(name), 'users.csv'), with_first_row=True)
    books, book_first = load_id_map(os.path.join(state_dir(name), 'books.csv'), with_first_row=True)
    user_perm = np.empty(len(users), dtype='int64')
    book_perm = np.empty(len(books), dtype='int64')
    new_users, new_books = {}, {}
    for new_id, raw in enumerate(sorted(users, key=lambda r: user_first[r])):
        user_perm[users[raw]] = new_id
        new_users[raw] = new_id
    for new_id, raw in enumerate(sorted(books, key=lambda r: book_first[r])):
        book_perm[books[raw]] = new_id
        new_books[raw] = new_id

    for f in sorted(glob.glob(os.path.join(state_dir(name), 'chunk_*.npz'))):
        part = np.load(f)
        user, item, time = user_perm[part['user']], book_perm[part['item']], part['time']
        order = np.argsort(user, kind='stable')  # keeps each user's time order
        np.savez(f, user=user[order], item=item[order], time=time[order])

    graph = dgl.load_graphs(graph_path(name))[0][0]
    total_edges = sum(len(np.load(f)['user']) for f in glob.glob(os.path.join(state_dir(name), 'chunk_*.npz')))
    if graph.num_edges('by') != total_edges:
        raise SystemExit('graph is missing chunks: run build-graph first')
    src, dst = graph.edges(etype='by')
    item = book_perm[src.numpy()]
    user = user_perm[dst.numpy()]
    time = graph.edges['by'].data['time'].numpy()
    # same edge order as new_data1.generate_graph: by book, then time, then user
    order = np.lexsort((user, time, item))
    user, item, time = user[order], item[order], time[order]
    graph = dgl.heterograph({('item', 'by', 'user'): (torch.tensor(item), torch.tensor(user)),
                             ('user', 'pby', 'item'): (torch.tensor(user), torch.tensor(item))},
                            num_nodes_dict={'user': len(users), 'item': len(books)})
    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)
    graph = set_node_ids(graph)
    save_graphs(graph_path(name), graph)

    save_id_map(new_users, user_first, os.path.join(state_dir(name), 'users.csv'))
    save_id_map(new_books, book_first, os.path.join(state_dir(name), 'books.csv'))
    open(os.path.join(state_dir(name), 'finalized'), 'w').close()
    print(f'finalized: {len(users):,} users, {len(books):,} books renumbered in original-file order; '
          f'graph relabelled -> {graph_path(name)}')


# ---------------------------------------------------------------------------------------------
# Sample generation. generate_user is new_data1.generate_user unchanged, except that it receives
# the user's own (time, item) sequence instead of scanning the whole DataFrame for it.

_graph = None


def _init_worker(path):
    global _graph
    torch.set_num_threads(1)
    _graph = dgl.load_graphs(path)[0][0]


def generate_user(user, u_time, u_seq, item_max_length, user_max_length, train_path, test_path, k_hop, val_path):
    graph = _graph
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    if len(u_seq) < 3:
        return 0, 0
    for j, t in enumerate(u_time[0:-1]):
        if j == 0:
            continue
        if j < item_max_length:
            start_t = u_time[0]
        else:
            start_t = u_time[j - item_max_length]
        sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
        sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
        sub_graph = dgl.edge_subgraph(graph, edges={'by': sub_u_eid, 'pby': sub_i_eid}, relabel_nodes=False)
        u_temp = torch.tensor([user])
        his_user = torch.tensor([user])
        graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
        i_temp = torch.unique(graph_i.edges(etype='by')[0])
        his_item = torch.unique(graph_i.edges(etype='by')[0])
        edge_i = [graph_i.edges['by'].data[dgl.NID]]
        edge_u = []
        for _ in range(k_hop-1):
            graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})
            u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
            his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
            i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
            his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
            edge_i.append(graph_i.edges['by'].data[dgl.NID])
            edge_u.append(graph_u.edges['pby'].data[dgl.NID])
        all_edge_u = torch.unique(torch.cat(edge_u))
        all_edge_i = torch.unique(torch.cat(edge_i))
        fin_graph = dgl.edge_subgraph(sub_graph, edges={'by': all_edge_i, 'pby': all_edge_u})
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


def _run_user(args):
    return generate_user(*args)


def generate_samples(name, chunk_numbers, item_max_length, user_max_length, k_hop, job):
    if not os.path.exists(graph_path(name)):
        raise FileNotFoundError(f'{graph_path(name)} not found: run add-chunk / build-graph first')
    total_edges = sum(len(np.load(f)['user']) for f in glob.glob(os.path.join(state_dir(name), 'chunk_*.npz')))
    graph_edges = dgl.load_graphs(graph_path(name))[0][0].num_edges('by')
    if graph_edges != total_edges:
        raise SystemExit(f'graph has {graph_edges:,} edges but the chunks have {total_edges:,}: '
                         f'run build-graph (or add-chunk without --no_graph_update) first')
    user, item, time = load_chunk_edges(name, chunk_numbers)
    # chunk files are already sorted by (user, time): split them into one sequence per user
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    ends = np.r_[starts[1:], len(user)]
    base = f'Newdata/{name}_{item_max_length}_{user_max_length}_{k_hop}'
    train_path, val_path, test_path = base + '/train/', base + '/val/', base + '/test/'
    tasks = [(int(user[s]), time[s:e], item[s:e], item_max_length, user_max_length,
              train_path, test_path, k_hop, val_path) for s, e in zip(starts, ends)]
    print(f'chunks {chunk_numbers or "all"}: {len(tasks):,} users  start: {datetime.datetime.now()}')
    with Pool(processes=job, initializer=_init_worker, initargs=(graph_path(name),)) as pool:
        all_num = pool.map(_run_user, tasks, chunksize=16)
    print('The number of train set:', sum(n[0] for n in all_num))
    print('The number of test set:', sum(n[1] for n in all_num))
    print('end:', datetime.datetime.now())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['add-chunk', 'build-graph', 'finalize', 'samples'])
    parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k_chunked', help='dataset name used for all outputs')
    parser.add_argument('--chunk', nargs='+', default=[], help='add-chunk: chunk CSV path(s), added in the given order')
    parser.add_argument('--no_graph_update', action='store_true', help='add-chunk: only update IDs/edges; build the graph later with build-graph')
    parser.add_argument('--chunks', type=int, nargs='*', default=None, help='samples: chunk numbers to generate (default: all)')
    parser.add_argument('--n_chunks', type=int, default=8, help='total number of chunks; samples refuse to run until all are added')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of worker processes')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    opt = parser.parse_args()
    check_name(opt.data)

    if opt.step == 'add-chunk':
        for path in opt.chunk:
            add_chunk(opt.data, path, update_graph=not opt.no_graph_update)
    elif opt.step == 'build-graph':
        build_graph(opt.data)
    elif opt.step == 'finalize':
        require_chunk_count = len(added_chunks(opt.data))
        if require_chunk_count != opt.n_chunks:
            raise SystemExit(f'{require_chunk_count} of {opt.n_chunks} chunks added; add all chunks first')
        finalize(opt.data)
    else:
        require_all_chunks(opt.data, opt.n_chunks)
        generate_samples(opt.data, opt.chunks, opt.item_max_length, opt.user_max_length, opt.k_hop, opt.job)
