import pandas as pd
import numpy as np
import torch

def generate_rotary_embedding(position, embedding_dim=50):
    """
    Generate rotary embeddings for a given position.
    
    Args:
        position (int): Position in sequence.
        embedding_dim (int): Dimension of embedding.
        
    Returns:
        list: Rotary embedding as a list of floats. 
    """
    theta = 10000 ** (-2 * (torch.arange(embedding_dim // 2, dtype=torch.float32) / embedding_dim))
    pos_enc = torch.cat([torch.sin(position * theta), torch.cos(position * theta)], dim=0)

    return pos_enc.tolist()

# Load dataset
df = pd.read_csv('Series_table_MovieLens_100k.csv')

# Helper function to extract series and position
def process_series(item_series_id):
    if item_series_id.startswith("SB"):  # Standalone item
        return "Standalone", -1
    elif "Series" in item_series_id:  # Series item
        parts = item_series_id.split("_")  # Split series identifier
        series_id = parts[0]  # e.g., "Series 1"
        position = int(parts[1][1:])  # Extract position (e.g., P5 -> 5)
        return series_id, position
    return None, None

# Create the mapping table
def create_series_table(df):
    records = []
    for _, row in df.iterrows():
        series_id, position = process_series(row["item_series_id"])
        standalone_or_series = "Standalone" if series_id == "Standalone" else "Series"
        records.append({
            "item_id": row["item_id"],
            "item_series_id": row["item_series_id"],
            "series_id": series_id,
            "position_in_series": position,
            "standalone_or_series": standalone_or_series
        })
    return pd.DataFrame(records)

series_table = create_series_table(df)

# Apply Rotary Embeddings Instead of Positional Encoding
embedding_dim = 50  # Change if needed
series_table['rotary_embedding'] = series_table.apply(
    lambda row: generate_rotary_embedding(row['position_in_series'], embedding_dim) 
                if row['position_in_series'] != -1 else [0.0] * embedding_dim, 
    axis=1
)

# Remove duplicates
series_table = series_table.drop_duplicates(subset=['item_id']).reset_index(drop=True)
series_table=series_table.drop('standalone_or_series',axis=1)
series_table = series_table[series_table['series_id'].str.startswith("Series", na=False)]

# Save generated series table with rotary embeddings
series_table.to_csv('Series_table_rotary_Series_table_MovieLens_100k.csv', index=False)