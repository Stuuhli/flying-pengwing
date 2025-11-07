#!/usr/bin/zsh

# This script starts the CLI service and then runs the python command to get the CLI

start() {
    echo "starting CLI..."
    export COMPUTE_MODE="NON_SLURM"
    apptainer exec --bind ./app:/app --bind ./CLI.py:/CLI.py fastapi_container.sif python /rwthfs/rz/cluster/home/rwth1751/PoCs/CLI.py
    sleep 5
    wait
}

# Call the start function
start

# Add trap to handle script termination
trap 'echo "Cleaning up..."; kill $(jobs -p) 2>/dev/null; sleep 2; exit' SIGINT EXIT
