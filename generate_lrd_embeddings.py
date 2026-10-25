import pandas as pd
import torch
import json
import gzip
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# 1. File Paths
dataset_name = 'Goodreads_young_adult_HSAL_100k'
interaction_file = f'Data/{dataset_name}.csv'
# Ensure you download the matching metadata file from the UCSD link
metadata_file = 'Data/goodreads_books_young_adult.json' 
output_tensor_file = f'Data/{dataset_name}_llm_embeddings.pt'

print("Loading interaction data and mimicking factorization...")
df = pd.read_csv(interaction_file)

# Mimic the EXACT factorization from new_data1.py and generate_neg.py
df['factorized_item_id'], unique_raw_ids = pd.factorize(df['item_id'])
item_num = len(unique_raw_ids)

# Create a mapping from the raw Goodreads book_id to the new factorized index
raw_to_factorized = {str(raw_id): fact_id for fact_id, raw_id in enumerate(unique_raw_ids)}

print(f"Total unique items to embed: {item_num}")

# 2. Extract Text from McAuley Metadata JSON
print("Parsing metadata JSON for text features...")
item_texts = [""] * item_num # Pre-allocate list to maintain index alignment
found_count = 0

with open(metadata_file, 'rt', encoding='utf-8') as f:
    for line in tqdm(f, desc="Scanning JSON"):
        record = json.loads(line)
        book_id = str(record.get('book_id', ''))
        
        # If this book is in our interaction dataset
        if book_id in raw_to_factorized:
            fact_id = raw_to_factorized[book_id]
            
            title = record.get('title', '')
            description = record.get('description', '')
            
            # Create a rich semantic string for the LLM
            text = f"Title: {title}. Description: {description}"
            item_texts[fact_id] = text
            found_count += 1

print(f"Matched text for {found_count} / {item_num} items.")

# 3. Generate LLM Embeddings
print("Loading SentenceTransformer model onto GPU...")
# This runs locally on your A100s
model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')

print("Generating language knowledge representations...")
# Encode in batches to maximize GPU throughput
embeddings = model.encode(item_texts, batch_size=256, show_progress_bar=True, convert_to_tensor=True)

# The output is [item_num, 384]. We keep it at 384 here.
# The LRD projection matrix W_1 will reduce this to your hidden_dim (50) inside the model.
print(f"Saving embedding tensor of shape {embeddings.shape} to {output_tensor_file}")
torch.save(embeddings.cpu(), output_tensor_file)
print("Done!")