#!/bin/bash

#SBATCH --job-name=cc-ex08-master
#SBATCH --output=master.out
#SBATCH --time=00:10:00             # Short time for master job
#SBATCH --partition=exercise-coco

gcc main.c -o benchmark -O0 -pthread -Wall -Wextra -pedantic

echo "avg_iterations,num_threads,insert_ratio,stream_size,duration,ops_per_sec" > benchmark.csv

for insert_ratio in 0.1 0.5 0.9; do
    for stream_size in 100000 500000 1000000 5000000 10000000 25000000 50000000; do
        for thread_count in 1 2 4 8 12 16 24 32 48; do
            sbatch --job-name="cc08-${stream_size}-${thread_count}-${insert_ratio}" \
                --time=04:00:00 \
                --partition=exercise-coco \
                --nodelist=csg-moore \
                --exclusive \
                --output="job_${stream_size}_${thread_count}_${insert_ratio}.out" \
                --error="job_${stream_size}_${thread_count}_${insert_ratio}.err" \
                --wrap="./benchmark $stream_size $thread_count $insert_ratio >> benchmark.csv"
        done
    done
done