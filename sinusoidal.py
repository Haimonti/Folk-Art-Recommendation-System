import pandas as pd
import numpy as np
import sys

def generate_positional_embedding(position, embedding_dim=10):
    """
    Generate a positional embedding for a given position.
    
    Args:
        position (int): The position of the item in the series.
        embedding_dim (int): The dimensionality of the embedding vector.
        
    Returns:
        list: A list representing the positional embedding.
    """
    # Example positional embedding (sinusoidal encoding)
    # embedding = [np.sin(position / (10000 ** (2 * i / embedding_dim))) if i % 2 == 0 
    #              else np.cos(position / (10000 ** (2 * i / embedding_dim))) 
    #              for i in range(embedding_dim)]
    embedding = [
        float(np.sin(position / (10000 ** (2 * (i // 2) / embedding_dim)))) if i % 2 == 0 
        else float(np.cos(position / (10000 ** (2 * (i // 2) / embedding_dim))))
        for i in range(embedding_dim)
    ]
              
    return embedding
# Convert to DataFrame
df = pd.read_csv('MovieLens_all.csv')
#df=df.dropna()
# Helper function to extract series and position
def process_series(item_series_id):
    # Handle NaN / None / non-string values safely
    if not isinstance(item_series_id, str):
        return "Standalone", -1

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

# Add a new column 'positional_embedding' to the DataFrame
embedding_dim = 50  # Change this to your desired embedding dimension
series_table['positional_embedding'] = series_table.apply(
    lambda row: generate_positional_embedding(row['position_in_series'], embedding_dim) 
                if row['position_in_series'] != -1 else [0.0] * embedding_dim, 
    axis=1
)

series_table = series_table.drop_duplicates(subset=['item_id']).reset_index(drop=True)

series_table=series_table.drop('standalone_or_series',axis=1)
series_table = series_table[series_table['series_id'].str.startswith("Series", na=False)]

series_table.to_csv('Series_table_sinusodial_Series_table_MovieLens_all.csv', index=False)
