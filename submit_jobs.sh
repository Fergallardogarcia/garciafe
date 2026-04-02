#!/bin/bash

# Find all the config files
files=(temp/CIFAR10/configs/more_samples/*more_samples_0*.yaml)

# Get the number of files
num_files=${#files[@]}

# If there are files, submit the job array
if [ "$num_files" -gt 0 ]; then
  # The array index is 0-based
  array_limit=$((num_files - 1))
  sbatch --array=0-$array_limit deploy_all.slurm
else
  echo "No config files found to run."
fi
