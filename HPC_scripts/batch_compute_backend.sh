#!/usr/bin/zsh
### Job name
#SBATCH --job-name=BACKEND
### Mention Project
#SBATCH --account=rwth1751
### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=/home/rwth1751/apptainer-job-%J.log
#SBATCH --error=/home/rwth1751/error-job-%J.log
### Request partition
#SBATCH --partition=c23g
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=06:00:00
### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=5G
### Request GPU
#SBATCH --gres=gpu:2
### Request number of CPUs
#SBATCH --cpus-per-task=16
### Change to the work directory
cd /rwthfs/rz/cluster/home/rwth1751/PoCs
### Execute the container
apptainer exec --nv --bind ./milvus_configs/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml --bind ./milvus_configs/user.yaml:/milvus/configs/user.yaml --bind ./milvus-data:/var/lib/milvus --bind ./etcd-data:/var/lib/milvus/etcd --bind ./milvus_configs/milvus.yaml:/milvus/configs/milvus.yaml milvus.v2.5.2.sif ./HPC_scripts/milvus_SBATCH.sh &
sleep 5



source ./HPC_scripts/HF_key.env

GPU_ID_GEN='0,1'
TENSOR_PARALLEL_SIZE_GEN=2
dtype="auto"
MAX_NUM_SEQS=2
LOG_FILE="./HPC_scripts/vllm_apptainer.log"
export BACKEND="vllm"
export COMPUTE_MODE="SLURM"
export MODEL="mistral3.2-q8_0"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
echo "Start generative model"
GEN_MODEL_PATH="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
GEN_TOKENIZER="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
GEN_SERVED_MODEL_NAME=$MODEL
GPU_MEMORY_UTILIZATION_GEN=0.8
MAX_GEN_LEN=32384
apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID_GEN --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env VLLM_LOGGING_LEVEL=INFO --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $GEN_MODEL_PATH --tokenizer_mode mistral --served-model-name $GEN_SERVED_MODEL_NAME --config_format mistral --load_format mistral\
    --max-model-len $MAX_GEN_LEN --tensor-parallel-size $TENSOR_PARALLEL_SIZE_GEN --swap-space 0 --dtype $dtype --quantization fp8 --enable-chunked-prefill --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_GEN \
    --max-num-seqs $MAX_NUM_SEQS --max-seq-len-to-capture $MAX_GEN_LEN --no-enforce-eager --port 8079 2>&1 | tee "$LOG_FILE" &

sleep 20

echo "start embedding models"
GPU_ID='0'
IFS=',' read -A GPU_ARRAY <<<"$GPU_ID"
typeset -p GPU_ARRAY
TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}
MODEL_PATH_EMBED_1="sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH_EMBED_2="Qwen/Qwen3-Embedding-0.6B"
SERVED_EMBED_NAME_1="allmini-22m-512"
SERVED_EMBED_NAME_2="qwen3_embed"
MAX_EMBED_LEN_1=512
MAX_EMBED_LEN_2=8192
dtype="auto"
MAX_NUM_SEQS=64
GPU_MEMORY_UTILIZATION=0.1 # default 0.9
apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $MODEL_PATH_EMBED_1 --trust-remote-code --served-model-name $SERVED_EMBED_NAME_1 --max-model-len $MAX_EMBED_LEN_1 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 --dtype $dtype --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-num-seqs $MAX_NUM_SEQS --port 8080 --task embedding 2>&1 | tee "$LOG_FILE" &

apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $MODEL_PATH_EMBED_2 --trust-remote-code --served-model-name $SERVED_EMBED_NAME_2 --max-model-len $MAX_EMBED_LEN_2 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 --dtype $dtype --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-num-seqs $MAX_NUM_SEQS --port 8081 --task embedding 2>&1 | tee "$LOG_FILE" &
sleep 10

echo "start reranking model"
GPU_MEMORY_UTILIZATION_RERANK=0.2
export MODEL_RERANK="BAAI/bge-reranker-v2-m3"
MODEL_PATH_RERANK=$MODEL_RERANK
SERVED_RERANKER=$MODEL_RERANK
MAX_RERANKER_LEN=8192
apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $MODEL_PATH_RERANK --trust-remote-code --served-model-name $SERVED_RERANKER --max-model-len $MAX_RERANKER_LEN --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 --dtype $dtype --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_RERANK --max-num-seqs $MAX_NUM_SEQS --port 8082 --task score 2>&1 | tee "$LOG_FILE" &
sleep 10

echo "start redis and fastapi services"
mkdir -p /rwthfs/rz/cluster/home/rwth1751/PoCs/user_data/doc_store
apptainer exec redis.sif redis-server &
apptainer exec --nv --bind /rwthfs/rz/cluster/home/rwth1751/PoCs/app:/app \
    --bind /rwthfs/rz/cluster/home/rwth1751/PoCs/user_data/doc_store:/doc_store \
    fastapi_container.sif \
    bash -c "cd /rwthfs/rz/cluster/home/rwth1751/PoCs && export BACKEND=vllm && uvicorn app.main:app --host 0.0.0.0 --port 8000" &
sleep 20

echo "Starting frontend"
apptainer exec --bind /rwthfs/rz/cluster/home/rwth1751/PoCs/frontend_gradio:/frontend_gradio gradio.sif python /frontend_gradio/frontend.py &


wait