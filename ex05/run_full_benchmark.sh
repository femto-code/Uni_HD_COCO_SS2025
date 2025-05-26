#!/bin/bash

gcc barrier_benchmark.c -o barrier_benchmark -O0 -pthread -lm
./barrier_benchmark pthread > pthread_results.csv
./barrier_benchmark counter > counter_results.csv
./barrier_benchmark dissemination > dissemination_results.csv
python3 plot.py