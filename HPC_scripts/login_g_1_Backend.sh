#!/usr/bin/zsh
# need to make above as instances
# This bash script starts 3 services together along with their shell commands 
# the 4 services are: Milvus DB, VLLM backend, FastAPI and Redis backend which has the RAG logic

export CUDA_VISIBLE_DEVICES=1
export OLLAMA_DEBUG=1
export CUDA_ERROR_LEVEL=50
export MODEL="mistral3.2-q8_0"
export BACKEND="vllm"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export MODEL_RERANK="BAAI/bge-reranker-v2-m3"
source ./HPC_scripts/HF_key.env


cleanup() {
    echo "Cleaning up"
    apptainer instance stop milvus_db
    sleep 2
    kill $(jobs -p) 2>/dev/null
    exit
}


start() {
    echo "Starting Milvus DB..."
    mkdir -p ./milvus-data
    mkdir -p ./etcd-data
    mkdir -p ./user_data
    apptainer instance start --bind ./milvus_configs/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml --bind ./milvus_configs/user.yaml:/milvus/configs/user.yaml --bind ./milvus-data:/var/lib/milvus --bind ./etcd-data:/var/lib/milvus/etcd --bind ./milvus_configs/milvus.yaml:/milvus/configs/milvus.yaml milvus.v2.5.2.sif milvus_db 
    timeout --signal=2 30m apptainer exec instance://milvus_db bash -c "export ETCD_USE_EMBED=true && export ETCD_DATA_DIR=/var/lib/milvus/etcd && export ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml && export COMMON_STORAGETYPE=local && cd /milvus && ./bin/milvus run standalone" &
    # Wait a few seconds for the server to start
    sleep 30

    # Load Generative model through vllm
    GPU_ID="1"
    IFS=',' read -r -a GPU_ARRAY <<< "$GPU_ID"
    # Calculate tensor parallel size based on the number of GPUs
    TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}
    dtype="auto"
    MAX_NUM_SEQS=16
    LOG_FILE="./HPC_scripts/vllm_apptainer.log"
    GEN_MODEL_PATH="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    GEN_TOKENIZER="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    GEN_SERVED_MODEL_NAME=$MODEL
    GPU_MEMORY_UTILIZATION_GEN=0.6
    MAX_GEN_LEN=16384
    echo "Loading generative model through vllm..."
    #timeout --signal=2 30m apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env VLLM_LOGGING_LEVEL=INFO --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    #--bind $PWD:/app vllm-openai_v0.9.1.sif --model $GEN_MODEL_PATH --tokenizer_mode mistral --served-model-name $GEN_SERVED_MODEL_NAME --config_format mistral --load_format mistral\
    #--max-model-len $MAX_GEN_LEN --tensor-parallel-size $TENSOR_PARALLEL_SIZE --swap-space 0 --dtype $dtype --quantization fp8 --enable-chunked-prefill --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_GEN \
    #--max-num-seqs $MAX_NUM_SEQS --port 8079 2>&1 | tee "$LOG_FILE" &
    #sleep 30

    # Load embedding model through vllm model
    MODEL_PATH_EMBED_1="sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH_EMBED_2="Qwen/Qwen3-Embedding-0.6B"
    SERVED_EMBED_NAME_1="allmini-22m-512"
    SERVED_EMBED_NAME_2="qwen3_embed"
    MAX_EMBED_LEN_1=512
    MAX_EMBED_LEN_2=16384
    GPU_MEMORY_UTILIZATION_EMBED=0.2 # default 0.9

    echo "Loading embedding model through vllm..."
    timeout --signal=2 30m apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $MODEL_PATH_EMBED_1 --trust-remote-code --served-model-name $SERVED_EMBED_NAME_1 --max-model-len $MAX_EMBED_LEN_1 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 --dtype $dtype --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_EMBED --max-num-seqs $MAX_NUM_SEQS --port 8080 --task embedding 2>&1 | tee "$LOG_FILE" &

    timeout --signal=2 30m apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $MODEL_PATH_EMBED_2 --trust-remote-code --served-model-name $SERVED_EMBED_NAME_2 --max-model-len $MAX_EMBED_LEN_2 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 --dtype $dtype --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_EMBED --max-num-seqs $MAX_NUM_SEQS --port 8081 --task embedding 2>&1 | tee "$LOG_FILE" &
    sleep 10

    # Load reranker model through vllm
    echo "Loading reranker model through vllm..."
    GPU_MEMORY_UTILIZATION_RERANK=0.2
    MODEL_PATH_RERANK=$MODEL_RERANK
    SERVED_RERANKER=$MODEL_RERANK
    MAX_RERANKER_LEN=8192
    timeout --signal=2 30m apptainer run --nv --env CUDA_VISIBLE_DEVICES=$GPU_ID --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN --env CC=gcc --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --bind $PWD:/app vllm-openai_v0.9.1.sif --model $MODEL_PATH_RERANK --trust-remote-code --served-model-name $SERVED_RERANKER --max-model-len $MAX_RERANKER_LEN --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --swap-space 0 --dtype $dtype --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_RERANK --max-num-seqs $MAX_NUM_SEQS --port 8082 --task score 2>&1 | tee "$LOG_FILE" &
    # Wait a few seconds for the server to start
    sleep 10
    # Start redis backend
    timeout --signal=2 30m apptainer exec redis.sif redis-server &
    sleep 5
    # Start FastAPI backend
    echo "Starting FastAPI backend..."
    mkdir -p ./user_data/doc_store
    timeout --signal=2 30m apptainer exec --nv --bind /rwthfs/rz/cluster/home/rwth1751/PoCs/app:/app \
    --bind /rwthfs/rz/cluster/home/rwth1751/PoCs/user_data/doc_store:/doc_store \
    fastapi_container.sif bash -c "cd /rwthfs/rz/cluster/home/rwth1751/PoCs && export BACKEND=vllm && uvicorn app.main:app --host 0.0.0.0 --port 8000" &
    
    # Wait for all background processes
    wait
}

# Add trap to handle script termination
trap cleanup SIGINT EXIT

start
