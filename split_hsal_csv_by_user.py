#!/usr/bin/env python
# Split one HSAL-format CSV (user_id,item_id,time,item_series_id) into N user-disjoint chunks.
# Every interaction of a user goes to the same chunk, so each chunk holds complete user histories.
import argparse
import os
import zlib

import pandas as pd


def shard_of(user_id, n):
    # stable across runs and machines (unlike Python's hash())
    return zlib.crc32(str(user_id).encode()) % n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k', help='CSV name inside ./Data (without .csv)')
    parser.add_argument('--n', type=int, default=8, help='number of chunks')
    parser.add_argument('--out_dir', default=None, help='default: Data/<data>_chunks')
    opt = parser.parse_args()

    out_dir = opt.out_dir or f'Data/{opt.data}_chunks'
    os.makedirs(out_dir, exist_ok=True)
    data = pd.read_csv(f'Data/{opt.data}.csv', dtype={'user_id': str, 'item_id': str})
    # row position in the original file: lets new_data1_chunked.py finalize give users/books the
    # same numbers as a single-file run
    data['orig_row'] = range(len(data))
    shard = data['user_id'].map(lambda u: shard_of(u, opt.n))
    for k in range(opt.n):
        part = data[shard == k]
        path = os.path.join(out_dir, f'{opt.data}_chunk_{k + 1}.csv')
        part.to_csv(path, index=False)
        print(f'{path}: {len(part):,} rows, {part["user_id"].nunique():,} users')
