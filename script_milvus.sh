#!/usr/bin/env bash

# This bash script starts 3 services together along with their shell commands 
# the 4 services are: Milvus DB, Ollama backend, FastAPI backend which has the RAG logic

declare -a PIDS

# Function to start a process and store its PID
start_process() {
    "$@" &
    local pid=$!
    PIDS+=($pid)
}

cleanup() {
    echo "Stopping milvus"
    apptainer instance stop milvus_db
    echo "All processes stopped."

    echo "Stopping all processes..."
    for pid in "${PIDS[@]}"; do
        # check if process exists and user has permission to send signals to it (-0), then kill and wait for the pid to be terminated
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            wait $pid 2>/dev/null
        fi
    done
    apptainer instance stop ollama
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
    MILVUS_INSTANCE_NAME="milvus_db"
    apptainer instance start --writable-tmpfs \
    --bind ./milvus_configs/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
    --bind ./milvus_configs/user.yaml:/milvus/configs/user.yaml \
    --bind ./milvus-data:/var/lib/milvus \
    --bind ./etcd-data:/var/lib/milvus/etcd \
    --bind ./milvus_configs/milvus.yaml:/milvus/configs/milvus.yaml \
    milvus.v2.5.2.sif "$MILVUS_INSTANCE_NAME"

    apptainer exec \
    instance://"$MILVUS_INSTANCE_NAME" \
    bash -c "export ETCD_USE_EMBED=true && \
    export ETCD_DATA_DIR=/var/lib/milvus/etcd && \
    export ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml && \
    export COMMON_STORAGETYPE=local && \
    cd /milvus && \
    ./bin/milvus run standalone" &
    sleep 30
    

    # Start Ollama serve in background
    echo "Starting Ollama server..."
    apptainer instance start ollama.sif ollama
    start_process apptainer exec instance://ollama ollama serve
    sleep 5
    
    # Load generative model
    echo "Loading generative model..."
    start_process apptainer exec instance://ollama ollama pull llama3.2:1b-instruct-q2_K
    sleep 5

    # Load embedding models
    echo "Loading embedding model..."
    #start_process apptainer exec instance://ollama ollama pull all-minilm:22m
    start_process apptainer exec instance://ollama bash -c 'echo -e "FROM all-minilm:22m\nPARAMETER num_ctx 512" > /tmp/Modelfile_allmini'
    apptainer exec instance://ollama ollama create allmini-22m-512 -f /tmp/Modelfile_allmini
    sleep 3
    #start_process apptainer exec instance://ollama bash -c 'echo -e "FROM snowflake-arctic-embed2\nPARAMETER num_ctx 8192" > /tmp/Modelfile_snowflake'
    #apptainer exec instance://ollama ollama create snowflake-arctic-embed2 -f /tmp/Modelfile_snowflake
    #sleep 5

    # Wait for all background processes

    if apptainer instance list | grep -q "$MILVUS_INSTANCE_NAME"; then
    echo "Instance '$MILVUS_INSTANCE_NAME' is running."
    else
        echo "Failed to start instance '$MILVUS_INSTANCE_NAME'."
        exit 1
    fi
    wait
}

# Call the start function
start


