# Chunked HSAL: building the graph chunk by chunk

These files build the HSAL graph from N user-disjoint chunks of a dataset instead of one big CSV,
so preprocessing can scale (GCP)

## Files added

**Core HSAL, chunked**

| New file | Based on | What changed |
|---|---|---|
| `split_hsal_csv_by_user.py` | (new) | Splits one HSAL CSV into N user-disjoint chunk CSVs; keeps each row's original position (`orig_row`) |
| `new_data1_chunked.py` | `new_data1.py` | Global user/book ID map, graph built chunk by chunk (`add-chunk`); `finalize` renumbers IDs in original-file order; samples generated per chunk of users (`samples --chunks k`) |
| `sinusoidal_chunked.py` | `sinusoidal.py` | Series table from all chunks with the global book IDs, parses `Series_<id>_P<position>` |
| `generate_neg_chunked.py` | `generate_neg.py` | Negatives from all chunks with the global IDs, book IDs stored as int32 (same values, half the size) |
| `new_main_BPR_chunked.py` | `new_main_BPR.py` | User/book counts from the ID map; `--seed`; `--resume`; uses `HSAL_for_BPR1_chunked.py` |
| `HSAL_for_BPR1_chunked.py` | `HSAL_for_BPR1.py` | Takes `data_name`, so it loads the series table of the dataset being trained |

**HSAL+LRD, chunked**

| New file | Based on | What changed |
|---|---|---|
| `HSAL_for_lrd_fixed.py` | `HSAL_for_lrd.py` | Fix: the LRD loss used the last book's position in the sample's subgraph (`last_alis`) as its book ID; it now converts it first (wrong for 256 of 256 tested examples before, right for 256 of 256 after). Takes `data_name` |
| `new_main_lrd_chunked.py` | `new_main_lrd.py` | Fixed model, counts from the ID map; `--seed`; `--resume`; file names tagged `_LRD_seed<k>` |
| `generate_llm_embeddings_v2.py` | `generate_lrd_embeddings.py` | Same text (`Title: ... Description: ...`), model (all-MiniLM-L6-v2) and row order; command-line options, `.json.gz` or stdin metadata, device choice, refuses to save if no book matches |
| `generate_llm_embeddings_chunked.py` | `generate_llm_embeddings_v2.py` | Row order from the chunked ID map (`books.csv`) |

**CCR job scripts**

`.sh` files contain shell commands and pipeline steps. `.sbatch` files tell CCR's SLURM scheduler how to run
a job, including resource requests and which script to execute.

| File | Note |
|---|---|
| `chunked_common.sh` | Shared settings for both jobs: folder, batch 256, seed 0, 10 epochs |
| `hsal_chunked_steps.sh` | HSAL pipeline: split, add chunks, finalize, series, negatives, samples, train |
| `lrd_chunked_steps.sh` | HSAL+LRD pipeline: waits for chunked preprocessing, checks embeddings, then trains |
| `run_hsal_chunked.sbatch` | Submits the HSAL experiment to CCR's SLURM job scheduler |
| `run_lrd_chunked.sbatch` | Submits the HSAL+LRD experiment to CCR's SLURM job scheduler |

Before using them on another account, change the folder in `chunked_common.sh` and the `-B` bind paths in the
two `.sbatch` files (they point at `/vscratch/grp-haimonti/snirwade` and `/projects/academic/haimonti`).

## Steps

```bash
# HSAL (chunked)
python split_hsal_csv_by_user.py --data Goodreads_Fantasy_HSAL_Final --n 8
for k in 1 2 3 4 5 6 7 8; do
  python new_data1_chunked.py add-chunk --data Goodreads_Fantasy_HSAL_Final_chunked \
      --chunk Data/Goodreads_Fantasy_HSAL_Final_chunks/Goodreads_Fantasy_HSAL_Final_chunk_$k.csv
done
python new_data1_chunked.py finalize --data Goodreads_Fantasy_HSAL_Final_chunked
python sinusoidal_chunked.py --data Goodreads_Fantasy_HSAL_Final_chunked
python generate_neg_chunked.py --data Goodreads_Fantasy_HSAL_Final_chunked
for k in 1 2 3 4 5 6 7 8; do   # can run as separate jobs, after ALL chunks are added
  python new_data1_chunked.py samples --data Goodreads_Fantasy_HSAL_Final_chunked --chunks $k
done
python new_main_BPR_chunked.py --data Goodreads_Fantasy_HSAL_Final_chunked --model_record --resume

# HSAL+LRD (chunked): reuses everything above, then
python generate_llm_embeddings_chunked.py --data Goodreads_Fantasy_HSAL_Final_chunked \
    --metadata Data/goodreads_books_fantasy_paranormal.json.gz
python new_main_lrd_chunked.py --data Goodreads_Fantasy_HSAL_Final_chunked --model_record --resume
```

On CCR (scavenger), one job per model and dataset:

```bash
sbatch --export=ALL,DATASET=Goodreads_Romance_HSAL_Final --job-name=HSALchunk_Romance --mem=128G run_hsal_chunked.sbatch
sbatch --export=ALL,DATASET=Goodreads_Romance_HSAL_Final --job-name=LRDchunk_Romance  --mem=128G run_lrd_chunked.sbatch
```

The HSAL+LRD job waits for the HSAL job's chunked preprocessing of the same dataset, then checks that the
book order matches the embedding rows before training. Its embeddings file
`Data/<dataset>_chunked_llm_embeddings.pt` must exist first; book metadata is the UCSD
`goodreads_books_<genre>.json.gz`.

Chunk CSVs must have the HSAL columns `user_id,item_id,time,item_series_id`.

## Rules

- Add every chunk before generating samples. A user's subgraph pulls in other users who read the
  same books, and those users can be in any chunk.
- IDs are assigned once, users in order of first appearance, books the first time they are seen.
  `Data/<name>_chunked/users.csv` and `books.csv` map raw IDs to model IDs.
- Adding the same chunk twice is skipped; a chunk containing users from an earlier chunk is rejected.
- `--data` must end with `_chunked`, so the chunked scripts never overwrite the original
  `Data/<name>_graph`, `<name>_neg`, series table or `Newdata/<name>_...` outputs.
- `finalize` must run after the last chunk and before samples, series table and negatives (they refuse
  otherwise). It refuses to run if any of those outputs already exist, since they would use old IDs.
- Sequels across chunks: a book read only in chunk 3 and its sequel read only in chunk 5 still get the
  same `series_id` in the series table, because the table is built from all chunks.

## Checks against the original single-file pipeline

- Young Adult test slice (12,000 users): identical users, books, all 36,850 graph edges, all 21,242
  samples (14,766 train / 2,652 val / 3,824 test), negatives and series table.
- Without `finalize`, about 15% of subgraphs differed: users were numbered chunk by chunk, and when more
  than 50 neighbour users are found `generate_user` keeps the 50 with the largest ID numbers.
  `finalize` removes the difference without changing that rule.
- Fantasy & Paranormal (599,097 rows): identical sample counts (254,807 / 38,998 / 57,463);
  preprocessing 8 min chunked vs about 7 h original; HSAL+LRD embeddings bit-identical for both pipelines.

**HSAL vs HSAL+LRD, original and chunked (epoch 10, Recall@10 / NDCG@10)**

| Dataset | Epoch compared | HSAL | HSAL chunked | HSAL+LRD | HSAL+LRD chunked | Difference (HSAL+LRD chunked − HSAL chunked) |
|---|---|---|---|---|---|---|
| Fantasy & Paranormal | 10 | 0.5391 / 0.3028 | 0.5422 / 0.3044 | 0.5437 / 0.3081 | 0.5466 / 0.3110 | +0.0044 / +0.0066 |
| Young Adult | 10 | ~0.383 / ~0.196 | 0.3826 / 0.1956 | ~0.394 / ~0.204 | 0.3939 / 0.2035 | +0.0113 / +0.0079 |
| Romance | 10 | ~0.485 / ~0.270 | ~0.485 / ~0.270 | ~0.496 / ~0.279 | 0.4964 / 0.2787 | ~+0.011 / ~+0.009 |
| Mystery, Thriller & Crime | 10 | ~0.457 / ~0.250 | ~0.457 / ~0.250 | ~0.481 / ~0.266 | 0.4808 / 0.2659 | ~+0.024 / ~+0.016 |


Chunked and original runs differ by about 0.003, which is run-to-run noise (one seed per run)
