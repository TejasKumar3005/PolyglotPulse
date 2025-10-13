#!/bin/bash

# Command line arguments
mode=$1
path_to_data=$2
path_to_save=$2

# Check if the mode is 'train' or 'test'
if [ "$mode" == "train" ]; then
    # Training mode
    python3 final.py train "$path_to_data" "$path_to_save"
elif [ "$mode" == "test" ]; then
    path_to_test_json=$3
    output_path=$4
    # Inference mode
    python3 final.py test "$path_to_save" "$path_to_test_json" "$output_path"
else
    echo "Invalid command. Use 'train' or 'test'."
fi
