import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("benchmark.csv", comment='#')

# Get all unique insert_ratio values
insert_ratios = sorted(df['insert_ratio'].unique())

for insert_ratio in insert_ratios:
    plt.figure(figsize=(10, 6))
    subset = df[df['insert_ratio'] == insert_ratio]
    for stream_size in sorted(subset['stream_size'].unique()):
        data = subset[subset['stream_size'] == stream_size]
        plt.plot(
            data['num_threads'],
            data['ops_per_sec'],
            marker='o',
            label=f"stream_size={stream_size}"
        )
    plt.xlabel("Number of Threads")
    plt.ylabel("Operations per Second")
    plt.title(f"Throughput vs Threads (insert_ratio={insert_ratio})")
    plt.legend(title="Stream Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ops_per_sec_insert_ratio_{insert_ratio}.svg")
    plt.close()

print("Plots saved as ops_per_sec_insert_ratio_*.svg")