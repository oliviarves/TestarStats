import numpy as np
import pandas as pd
from scipy.stats import kruskal, bootstrap
from pingouin import pairwise_tukey, compute_effsize
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# Data corresponding to 1 app for all tools
data = {
    'T1_A1': [31, 26, 23, 28, 28, 26, 25, 25, 19, 25],
    'T2_A1': [35, 28, 26, 29, 32, 30, 28, 27, 22, 30],
    'T3_A1': [34, 30, 28, 32, 33, 31, 30, 29, 25, 31],
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Kruskal-Wallis test
statistic, p_value = kruskal(df['T1_A1'], df['T2_A1'], df['T3_A1'])
print(f"Kruskal-Wallis Test:")
print(f"Statistic: {statistic}")
print(f"P-value: {p_value}")

# Post-hoc test (pairwise Tukey)
posthoc_tukey = pairwise_tukey(data=df.melt(value_vars=df.columns), dv='value', between='variable')
print("\nPost-hoc (Tukey) Test:")
print(posthoc_tukey)

# Effect size calculation (Cohen's d)
effect_size = compute_effsize(df['T1_A1'], df['T2_A1'], 'CLES')
print(f"\nEffect Size (Cohen's d) df['T1_A1'], df['T2_A1']: {effect_size}")
effect_size = compute_effsize(df['T1_A1'], df['T3_A1'], 'CLES')
print(f"\nEffect Size (Cohen's d) df['T1_A1'], df['T3_A1']: {effect_size}")
effect_size = compute_effsize(df['T2_A1'], df['T3_A1'], 'CLES')
print(f"\nEffect Size (Cohen's d) df['T2_A1'], df['T3_A1']: {effect_size}")

# Bootstrapping for median difference
def median_difference(data1, data2):
    return np.median(data1) - np.median(data2)

# Perform bootstrap resampling
bootstrap_results = bootstrap((df['T1_A1'], df['T2_A1']))

# Extract bootstrap results
confidence_interval = np.percentile(bootstrap_results, [2.5, 97.5])
