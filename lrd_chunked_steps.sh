# HSAL+LRD, chunked: reuses the chunked preprocessing made by hsal_chunked_steps.sh for the same DATASET.
# If that job is still preprocessing, wait for it (up to 4 h) instead of building the same files twice.
source chunked_common.sh
step "dataset $DATASET"
for i in $(seq 1 240); do
    [ -f markers/${NAME}_prep_done ] && break
    [ $i -eq 1 ] && step "waiting for the HSAL chunked job to finish preprocessing $NAME"
    sleep 60
done
[ -f markers/${NAME}_prep_done ] || { step "preprocessing for $NAME never finished"; exit 1; }
[ -f Data/${NAME}_llm_embeddings.pt ] || { step "Data/${NAME}_llm_embeddings.pt missing"; exit 1; }
# the embeddings were made with the single-file numbering; finalize must give exactly the same book order
python - <<PYEOF
import pandas as pd, torch
csv_order = list(pd.factorize(pd.read_csv('Data/${DATASET}.csv', dtype={'item_id': str})['item_id'])[1])
books = pd.read_csv('Data/${NAME}_chunked/books.csv', dtype={'raw_id': str}).sort_values('id')
emb = torch.load('Data/${NAME}_llm_embeddings.pt')
assert list(books['raw_id']) == csv_order, 'book order of the chunked ID map differs from the embeddings'
assert emb.shape[0] == len(books), f'embeddings have {emb.shape[0]} rows, ID map has {len(books)} books'
print(f'embedding check passed: {emb.shape[0]} books in the same order')
PYEOF
step "training HSAL+LRD (bug-fixed, seeded)"
python -u new_main_lrd_chunked.py --data $NAME $TRAIN_ARGS
step "done"
