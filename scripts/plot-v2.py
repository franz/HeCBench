#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

outlier_threshold = 0.5

def read_csv(filename, suffix):
    df = pd.read_csv(filename, header=None, names=['benchmark', 'min', 'mean', 'std', 'cv'])
    df['benchmark'] = df['benchmark'].str.replace(suffix, '')
    return df.set_index('benchmark')

def geometric_mean(series):
    return np.exp(np.mean(np.log(series)))

def geometric_mean_without_outliers(series, threshold=outlier_threshold):
    log_series = np.log(series)
    return np.exp(np.mean(log_series[(log_series > np.log(1-threshold)) & (log_series < np.log(1+threshold))]))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate benchmark comparison plot.')
parser.add_argument('-b', '--baseline', required=True, nargs=2, metavar=('FILE', 'LABEL'), help='Baseline CSV file and label')
parser.add_argument('-c', '--comparison', action='append', required=True, nargs=2, metavar=('FILE', 'LABEL'), help='Comparison CSV file and label (can be used multiple times)')
parser.add_argument('-o', '--output', required=True, help='Output filename for the plot')
args = parser.parse_args()

# Read baseline file
baseline_file, baseline_label = args.baseline
sycl_df = read_csv(baseline_file, '-sycl')

# Read comparison file(s)
comparison_dfs = [read_csv(file, '-hip') for file, _ in args.comparison]
comparison_labels = [label for _, label in args.comparison]

# Find common benchmarks
common_benchmarks = sycl_df.index
for df in comparison_dfs:
    common_benchmarks = common_benchmarks.intersection(df.index)

# Filter dataframes to include only common benchmarks
sycl_df = sycl_df.loc[common_benchmarks]
comparison_dfs = [df.loc[common_benchmarks] for df in comparison_dfs]

# Create the main dataframe
df = pd.DataFrame({
    'Benchmark': common_benchmarks,
    baseline_label: sycl_df['mean']
})

# Add comparison data and calculate speedups
for i, (comp_df, label) in enumerate(zip(comparison_dfs, comparison_labels)):
    df[label] = comp_df['mean']
    df[f'{label}_Speedup'] = df[baseline_label] / df[label]

# Sort by the first speedup column
first_speedup_col = f'{comparison_labels[0]}_Speedup'
df = df.sort_values(first_speedup_col, ascending=True)

# Plotting
plt.figure(figsize=(20, 12))
sns.set(style="whitegrid")

x = range(len(df))
width = 0.8 / len(comparison_dfs)

colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(comparison_dfs)))

for i, (comp_df, label) in enumerate(zip(comparison_dfs, comparison_labels)):
    speedup_col = f'{label}_Speedup'
    plt.bar([j + i*width for j in x], df[speedup_col], width, label=label, color=colors[i])
    
    for j, speedup in enumerate(df[speedup_col]):
        plt.text(j + i*width, speedup, f'{speedup:.2f}x', ha='center', va='bottom', rotation=90)

plt.xlabel('Benchmarks', fontsize=12)
plt.ylabel(f'Speedup relative to {baseline_label} (higher is better)', fontsize=12)
plt.title(f'Speedup Relative to {baseline_label}', fontsize=16)
plt.xticks([i + width*(len(comparison_dfs)-1)/2 for i in x], df['Benchmark'], rotation=90, ha='right')
plt.legend()

plt.axhline(y=1, color='r', linestyle='--', label=f'{baseline_label} Baseline')

plt.yscale('log')

# Calculate and display geometric means
geomean_text = f'Outlier threshold: {outlier_threshold}\nGeo Mean Speedup (with/without outliers):\n'
for label in comparison_labels:
    speedup_col = f'{label}_Speedup'
    geo_mean = geometric_mean(df[speedup_col])
    geo_mean_no_outliers = geometric_mean_without_outliers(df[speedup_col])
    geomean_text += f'{label}: {geo_mean:.2f}x / {geo_mean_no_outliers:.2f}x\n'

plt.text(0.95, 0.95, geomean_text, 
         transform=plt.gca().transAxes, ha='right', va='top', 
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.tight_layout()
plt.savefig(args.output, dpi=300)
plt.close()

print(f"Plot saved as '{args.output}'")
print(f"Number of common benchmarks: {len(common_benchmarks)}")