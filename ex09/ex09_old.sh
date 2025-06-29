#!/bin/bash
#SBATCH --job-name=cc-ex08           # Job name
#SBATCH --output=job.out             # Standard output and error log
#SBATCH --error=job.err              # Error log
#SBATCH --time=04:00:00              # Time limit hrs:min:sec
#SBATCH --partition=exercise-coco    # Partition name
#SBATCH --nodelist=csg-moore         # Specific node
#SBATCH --exclusive                  # Exclusive node for benchmarking
#SBATCH --cpus-per-task=48           # Reserve 48 CPUs for this task

gcc main.c -o benchmark -O0
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
./benchmark
python3 plot.py