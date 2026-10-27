# generate_llm_embeddings.py with command-line options (same text, model and row order).
# Row i of Data/<data>_llm_embeddings.pt = book id i, numbered by pd.factorize(item_id) on the CSV,
# exactly as in new_data1.py and generate_neg.py.
#   --metadata  Goodreads books file: .json or .json.gz, or '-' to read it from stdin
#               (e.g.  ssh ccr 'cat Data/goodreads_books_fantasy_paranormal.json.gz' | python ... --metadata -)
import argparse
import gzip
import io
import json
import sys

import pandas as pd
import torch
from tqdm import tqdm


def open_metadata(path):
    raw = sys.stdin.buffer if path == '-' else open(path, 'rb')
    raw = io.BufferedReader(raw) if path == '-' else raw
    if raw.peek(2)[:2] == b'\x1f\x8b':          # gzip magic number
        return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding='utf-8')
    return io.TextIOWrapper(raw, encoding='utf-8')


def book_texts(metadata_path, raw_to_index, n_items):
    """Text per row, built exactly like generate_llm_embeddings.py ('' if the book has no metadata)."""
    item_texts = [""] * n_items
    found_count = 0
    with open_metadata(metadata_path) as f:
        for line in tqdm(f, desc="Scanning JSON"):
            record = json.loads(line)
            book_id = str(record.get('book_id', ''))
            if book_id in raw_to_index:
                title = record.get('title', '')
                description = record.get('description', '')
                item_texts[raw_to_index[book_id]] = f"Title: {title}. Description: {description}"
                found_count += 1
    print(f"Matched text for {found_count} / {n_items} items.")
    if found_count == 0:
        raise SystemExit('no book in the metadata matched the dataset: wrong or empty --metadata, nothing saved')
    return item_texts


def embed_and_save(item_texts, output_file, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embeddings = model.encode(item_texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    print(f"Saving embedding tensor of shape {tuple(embeddings.shape)} to {output_file}")
    torch.save(embeddings.cpu(), output_file)


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Goodreads_Fantasy_HSAL_Final', help='CSV name inside ./Data (without .csv)')
    parser.add_argument('--metadata', default='Data/goodreads_books_fantasy_paranormal.json.gz')
    parser.add_argument('--device', default=None, help='cuda / mps / cpu (default: best available)')
    opt = parser.parse_args()

    df = pd.read_csv(f'Data/{opt.data}.csv')
    _, unique_raw_ids = pd.factorize(df['item_id'])          # same numbering as new_data1.py
    raw_to_index = {str(raw_id): i for i, raw_id in enumerate(unique_raw_ids)}
    print(f"Total unique items to embed: {len(unique_raw_ids)}")
    texts = book_texts(opt.metadata, raw_to_index, len(unique_raw_ids))
    embed_and_save(texts, f'Data/{opt.data}_llm_embeddings.pt', opt.device or default_device())
