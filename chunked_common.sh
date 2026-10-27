# Settings shared by the generic chunked jobs (same values as fantasy_common.sh).
# DATASET comes from the job: sbatch --export=ALL,DATASET=Goodreads_Romance_HSAL_Final ...
set -eo pipefail
cd /vscratch/grp-haimonti/snirwade/HSAL-Chunking
source venv/bin/activate
export LD_LIBRARY_PATH='/.singularity.d/libs'
mkdir -p save_models save_checkpoints markers logs
: "${DATASET:?set DATASET, e.g. sbatch --export=ALL,DATASET=Goodreads_Romance_HSAL_Final ...}"
NAME=${DATASET}_chunked
N_CHUNKS=8
ITEM_LEN=50
USER_LEN=50
KHOP=2
EPOCHS=10
SEED=0
TRAIN_ARGS="--batchSize 256 --item_max_length $ITEM_LEN --user_max_length $USER_LEN --k_hop $KHOP --lr 0.001 --l2 0.0001 --layer_num 3 --hidden_size 50 --gpu 0 --seed $SEED --epoch $EPOCHS --model_record --resume"
step() { echo "[$(date +%F' '%T)] $*"; }
