#!/bin/bash

# Array of dataset folders
datasets=(
  "all_phases"
  "Phase_1"
  "Phase_2"
  "Phase_3"
  "Phase_4"
  "Phase_5"
  "Phase_6"
  "Phase_7"
  "Phase_8"
  "Phase_9"
  "Phase_10"
)


# # Array of dataset folders
# datasets=(
#   "all_phases"
# )
# #   "Phase_1"
# #   "Phase_2"
# #   "Phase_3"
# #   "Phase_4"
# #   "Phase_5"
# #   "Phase_6"
# #   "Phase_7"
# #   "Phase_8"
# #   "Phase_9"
# #   "Phase_10"
# # )

# Model and hyperparameters
model="LSTM"
batch_size=8
device_num=0

# Output directory
output_dir="results"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each dataset
for dataset in "${datasets[@]}"
do
  echo "Running model on dataset: $dataset"
  
  # Run the Python script with the specified arguments
  python main.py --root_dir "data/$dataset" --model "$model" --batch_size "$batch_size" --device_num "$device_num" --label $dataset
  
  echo "Finished running model on dataset: $dataset"
  echo "------------------------------------------------"
done

echo "All runs completed."