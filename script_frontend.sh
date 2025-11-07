#!/usr/bin/env bash

# This script starts the CLI service and then runs the python command to get the CLI

cleanup() {
    echo "Stopping frontend processes..."
    apptainer instance stop frontend
    trap - EXIT
    exit 0
}

# Add trap to handle script termination
trap cleanup SIGINT EXIT

start_CLI() {
    echo "starting CLI..."
    apptainer exec --bind ./app:/app --bind ./CLI.py:/CLI.py instance://fastapi_server python CLI.py
    sleep 5
    wait
}

start_gradio() {
    echo "Startig Gradio frontend"
    apptainer instance start gradio.sif frontend
    export COMPUTE_MODE="NON_SLURM"
    apptainer exec --bind ./frontend_gradio:./frontend_gradio instance://frontend python frontend_gradio/frontend.py

    wait
}

# Call the start function
start_gradio