#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if a model file was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_audit.sh <model_file.pkl>"
    echo "Example: ./run_audit.sh uploads/credit_random_model.pkl"
    exit 1
fi

# Run the audit
echo "Running audit on $1..."
python3 src/run_audit_cli.py "$1" 