# Core HSAL, chunked: same steps as fantasy_chunked_steps.sh for any HSAL CSV in Data/
source chunked_common.sh
step "dataset $DATASET"
if [ -f markers/${NAME}_prep_done ]; then
    step "preprocessing already done, skipping"
else
    # a stopped job may have left a half-built chunked state: start it from scratch
    rm -rf Data/${DATASET}_chunks Data/${NAME}_chunked Data/${NAME}_graph ${NAME}_neg Series_table_sinusodial_Series_table_${NAME}.csv Newdata/${NAME}_${ITEM_LEN}_${USER_LEN}_${KHOP}
    step "split into $N_CHUNKS user-disjoint chunks"
    python split_hsal_csv_by_user.py --data $DATASET --n $N_CHUNKS
    for k in $(seq 1 $N_CHUNKS); do
        step "add chunk $k"
        python new_data1_chunked.py add-chunk --data $NAME --chunk Data/${DATASET}_chunks/${DATASET}_chunk_$k.csv
    done
    step "finalize";      python new_data1_chunked.py finalize --data $NAME --n_chunks $N_CHUNKS
    step "series table";  python sinusoidal_chunked.py --data $NAME --n_chunks $N_CHUNKS
    step "negatives";     python generate_neg_chunked.py --data $NAME --n_chunks $N_CHUNKS | grep -E "GB of|written"
    for k in $(seq 1 $N_CHUNKS); do
        step "samples for chunk $k"
        python new_data1_chunked.py samples --data $NAME --chunks $k --n_chunks $N_CHUNKS --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --job 16
    done
    touch markers/${NAME}_prep_done
    step "preprocessing done"
fi
step "training HSAL"
python -u new_main_BPR_chunked.py --data $NAME $TRAIN_ARGS
step "done"
