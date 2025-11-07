#!/usr/bin/env bash

# This bash script starts 3 services together along with their shell commands 
# the 4 services are: Milvus DB, Ollama backend, FastAPI backend which has the RAG logic

cleanup() {
    echo "Stopping milvus"
    apptainer instance stop milvus_db
    echo "All processes stopped."
    exit 0
}

# Add trap to handle script termination
trap cleanup SIGINT EXIT

start() {
    # Start Milvus DB
    # Kill process on port 2380
    lsof -ti tcp:2380 | xargs kill -9
    # Wait a moment to ensure the port is released
    sleep 2
    echo "Starting Milvus DB..."
    # First create separate directories
    mkdir -p ./milvus-data
    mkdir -p ./etcd-data
    mkdir -p ./user_data
    apptainer instance start --writable-tmpfs \
    --bind ./milvus_configs/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
    --bind ./milvus_configs/user.yaml:/milvus/configs/user.yaml \
    --bind ./milvus-data:/var/lib/milvus \
    --bind ./etcd-data:/var/lib/milvus/etcd \
    --bind ./milvus_configs/milvus.yaml:/milvus/configs/milvus.yaml \
    milvus.v2.5.2.sif milvus_db

    apptainer exec \
    instance://milvus_db \
    bash -c "export ETCD_USE_EMBED=true && \
    export ETCD_DATA_DIR=/var/lib/milvus/etcd && \
    export ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml && \
    export COMMON_STORAGETYPE=local && \
    cd /milvus && \
    ./bin/milvus run standalone"
    sleep 10
    
    apptainer instance start redis.sif redis_backend
    apptainer exec instance://redis_backend redis-server
    sleep 5
    # Wait for all background processes
    wait
}

# Call the start function
start


