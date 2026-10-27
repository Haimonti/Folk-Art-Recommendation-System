# Chunked version of generate_llm_embeddings_v2.py: row i = book id i in the global ID map
# (Data/<data>_chunked/books.csv, written by new_data1_chunked.py), so no single CSV is needed.
# Run after `new_data1_chunked.py finalize`; text, model and batch size are unchanged.
import argparse
import os

import pandas as pd

from generate_llm_embeddings_v2 import book_texts, default_device, embed_and_save


# same rules as new_data1_chunked.py (copied so this step does not need DGL)
def state_dir(name):
    return f'Data/{name}_chunked'


def check_name(name):
    if not name.endswith('_chunked'):
        raise SystemExit(f"--data '{name}' must end with '_chunked' so original outputs are never overwritten")


def is_finalized(name):
    return os.path.exists(os.path.join(state_dir(name), 'finalized'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Goodreads_Fantasy_HSAL_Final_chunked', help='dataset name used by new_data1_chunked.py')
    parser.add_argument('--metadata', default='Data/goodreads_books_fantasy_paranormal.json.gz')
    parser.add_argument('--device', default=None, help='cuda / mps / cpu (default: best available)')
    opt = parser.parse_args()
    check_name(opt.data)
    if not is_finalized(opt.data):
        raise SystemExit(f'{opt.data} is not finalized: book ids are not final yet')

    books = pd.read_csv(os.path.join(state_dir(opt.data), 'books.csv'), dtype={'raw_id': str})
    if sorted(books['id']) != list(range(len(books))):
        raise SystemExit('books.csv ids are not 0..N-1')
    raw_to_index = dict(zip(books['raw_id'], books['id']))
    print(f"Total unique items to embed: {len(books)}")
    texts = book_texts(opt.metadata, raw_to_index, len(books))
    embed_and_save(texts, f'Data/{opt.data}_llm_embeddings.pt', opt.device or default_device())
