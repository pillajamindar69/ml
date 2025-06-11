import statistics
import numpy as np

# Sample dataset
data = [10, 12, 23, 23, 16, 23, 21, 16, 18, 20]

mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)
variance = statistics.variance(data)
std_dev = statistics.stdev(data)

print("--- Central Tendency Measures ---")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")

print("\n--- Dispersion Measures ---")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")