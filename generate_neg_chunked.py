#!/usr/bin/env python
# Chunked version of generate_neg.py: same output format (<name>_neg, a pickled pandas Series of
# never-read book ids per user), but uses the global IDs from new_data1_chunked.py and reads the
# chunks one at a time.
#
# Like the original, this stores every unread book for every user (users x books values); chunking
# does not reduce that. Book ids are stored as int32 instead of int64 (same values, half the memory);
# the training code converts them with .long().
import argparse
import os
import pickle

import numpy as np
import pandas as pd

from new_data1_chunked import check_name, require_all_chunks, load_chunk_edges, load_id_map, state_dir, added_chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k_chunked', help='dataset name used by new_data1_chunked.py')
    parser.add_argument('--n_chunks', type=int, default=8, help='total number of chunks')
    opt = parser.parse_args()
    check_name(opt.data)
    require_all_chunks(opt.data, opt.n_chunks)

    n_users = len(load_id_map(os.path.join(state_dir(opt.data), 'users.csv')))
    n_books = len(load_id_map(os.path.join(state_dir(opt.data), 'books.csv')))
    if n_books > np.iinfo(np.int32).max:
        raise SystemExit('book ids do not fit in int32')
    print(f'{n_users:,} users x {n_books:,} books: about {n_users * n_books * 4 / 1e9:.1f} GB of negatives')

    all_items = np.arange(n_books, dtype=np.int32)
    neg = {}
    for k in range(1, len(added_chunks(opt.data)) + 1):
        user, item, _ = load_chunk_edges(opt.data, [k])
        starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
        ends = np.r_[starts[1:], len(user)]
        for s, e in zip(starts, ends):
            neg[int(user[s])] = np.setdiff1d(all_items, item[s:e].astype(np.int32))
        print(f'chunk {k}: {len(starts):,} users')
    data_neg = pd.Series(neg).sort_index()
    data_neg.index.name = 'user_id'

    with open(opt.data + '_neg', 'wb') as f:
        pickle.dump(data_neg, f)
    print(f'{opt.data}_neg written ({len(data_neg):,} users)')
