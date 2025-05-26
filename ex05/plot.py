import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('pthread_results.csv')
df2 = pd.read_csv('counter_results.csv')
df3 = pd.read_csv('dissemination_results.csv')

plt.plot(df1['num_threads'], df1['avg_latency_us'], label='pthread Barrier')
plt.plot(df2['num_threads'], df2['avg_latency_us'], label='Counter Barrier')
plt.plot(df3['num_threads'], df3['avg_latency_us'], label='Dissemination Barrier')
plt.xlabel("Number of Threads")
plt.ylabel("Avg Barrier Latency (µs)")
plt.title("Barrier Latency Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("barrier_perf_comparison_plot.png")
plt.show()
