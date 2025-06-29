# plot_rbt_bench.py
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("job.out")

# Plot: ops_per_sec vs num_threads for each stream_size
plt.figure(figsize=(10, 6))
for stream_size in sorted(df['stream_size'].unique()):
    subset = df[df['stream_size'] == stream_size]
    plt.plot(subset['num_threads'], subset['ops_per_sec'], marker='o', label=f"stream_size={stream_size}")

plt.xlabel("Number of Threads")
plt.ylabel("Operations per Second")
plt.title("RBT Concurrent Benchmark: Throughput vs Threads")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("moore.svg")
plt.show()