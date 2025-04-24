#!/bin/bash

# Set CONDA_CHANNELS environment variable
export CONDA_CHANNELS=conda-forge

# Run the flow with kubernetes
python trainingflowgcp.py --environment=conda run --with kubernetes