#!/usr/bin/env bash

# Array to store PIDs of processes started by this script
declare -a PIDS

# Function to handle cleanup
cleanup() {
    echo "Stopping all processes..."
    for pid in "${PIDS[@]}"; do
        # check if process exists and user has permission to send signals to it (-0), then kill and wait for the pid to be terminated
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            wait $pid 2>/dev/null
        fi
    done
    apptainer instance stop fastapi_server
    apptainer instance stop redis_backend
    echo "All processes stopped."
    exit 0
}

# Trap SIGINT and EXIT signals
trap cleanup SIGINT EXIT

# Function to start a process and store its PID
start_process() {
    "$@" &
    local pid=$!
    PIDS+=($pid)
}

start() {
    

    # Start Redis backend
    echo "Starting Redis backend..."
    apptainer instance start redis.sif redis_backend
    start_process apptainer exec instance://redis_backend redis-server
    sleep 5
    mkdir -p ./user_data/doc_store
    # Start FastAPI backend
    echo "Starting FastAPI backend..."
    export MODEL="llama3.2:1b-instruct-q2_K"
    export BACKEND="ollama"
    apptainer instance start \
    --bind ./app:/app \
    --bind ./user_data/doc_store:/doc_store \
    fastapi_container.sif fastapi_server
    start_process apptainer exec instance://fastapi_server uvicorn app.main:app --host 0.0.0.0 --port 8000
    sleep 5

    # Wait for all background processes
    wait
}

# Call the start function
start
