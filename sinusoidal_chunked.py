#!/usr/bin/env python
# Chunked version of sinusoidal.py: builds the series/position table from all chunk CSVs,
# using the global book IDs written by new_data1_chunked.py (Data/<name>_chunked/books.csv)
# instead of renumbering books from one CSV.
import argparse
import os

import pandas as pd

from new_data1_chunked import check_name, require_all_chunks, added_chunks, load_id_map, state_dir


def generate_positional_embedding(position, embedding_dim=10):
    import numpy as np
    return [
        float(np.sin(position / (10000 ** (2 * (i // 2) / embedding_dim)))) if i % 2 == 0
        else float(np.cos(position / (10000 ** (2 * (i // 2) / embedding_dim))))
        for i in range(embedding_dim)
    ]


def process_series(item_series_id):
    if not isinstance(item_series_id, str):
        return "Standalone", -1
    if item_series_id.startswith("SB"):
        return "Standalone", -1
    elif item_series_id.startswith("Series_"):  # Series_<id>_P<position>
        parts = item_series_id.split("_")
        series_id = f"{parts[0]}_{parts[1]}"
        position = int(parts[2][1:])
        return series_id, position
    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Goodreads_young_adult_HSAL_100k_chunked', help='dataset name used by new_data1_chunked.py')
    parser.add_argument('--n_chunks', type=int, default=8, help='total number of chunks')
    parser.add_argument('--embedding_dim', type=int, default=50)
    opt = parser.parse_args()
    check_name(opt.data)
    require_all_chunks(opt.data, opt.n_chunks)

    books = load_id_map(os.path.join(state_dir(opt.data), 'books.csv'))
    # one row per book is enough: a book's series id is the same in every interaction
    parts = []
    for path in added_chunks(opt.data):
        df = pd.read_csv(path, dtype={'user_id': str, 'item_id': str}, usecols=['item_id', 'item_series_id'])
        parts.append(df.drop_duplicates(subset=['item_id']))
    df = pd.concat(parts).drop_duplicates(subset=['item_id']).reset_index(drop=True)
    df['item_id'] = df['item_id'].map(books)

    parsed = df['item_series_id'].map(process_series)
    series_table = pd.DataFrame({
        'item_id': df['item_id'],
        'item_series_id': df['item_series_id'],
        'series_id': parsed.map(lambda p: p[0]),
        'position_in_series': parsed.map(lambda p: p[1]),
    })
    series_table = series_table[series_table['series_id'].str.startswith("Series", na=False)].copy()
    series_table['positional_embedding'] = series_table['position_in_series'].map(
        lambda pos: generate_positional_embedding(pos, opt.embedding_dim))

    out = f'Series_table_sinusodial_Series_table_{opt.data}.csv'
    series_table.to_csv(out, index=False)
    print(f'{out}: {len(series_table):,} series books')
