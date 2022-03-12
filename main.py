from scipy.stats import mannwhitneyu

from process_data import get_data_from_csv, get_columns

# data = get_data_from_csv()
# technique1 = get_columns(data, 200, 'random')
# technique2 = get_columns(data, 200, 'abt')

technique1 = [41.52, 50.75, 44.7, 54.91, 55.89, 55.89, 41.18, 38.17, 53.26, 53.13, 44, 44.45, 34.08, 45.63, 49.24,
              41.18, 39.12, 44.67, 42.58, 57.24, 41.46, 48.73, 46.55, 49.38, 51.02, 54.46, 44.2, 45.55, 55.16, 44.04]
technique2 = [47.01, 46.6, 46.88, 45.24, 44.83, 55.39, 45.77, 44.41, 54.67, 57.33, 56.79, 48.78, 45.4, 55.61, 47.47,
              45.58, 56.17, 45.39, 53.91, 52.77, 54.62, 58.52, 55.53, 56.44, 47.74, 47.34, 55.39, 55.43, 56.63, 44.82]

print(technique1)
print(technique2)

import numpy as np

print([np.var(x, ddof=1) for x in [technique1, technique2]])

from matplotlib import pyplot as plt

x = np.arange(1, len(technique1) + 1)
plt.hist(technique1, color='blue', edgecolor='black', bins=6)

# seaborn histogram
# sns.distplot(flights['arr_delay'], hist=True, kde=False,
#              bins=int(180/5), color = 'blue',
#              hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Technique 1')
plt.show()

x = np.arange(1, len(technique2) + 1)
plt.hist(technique2, color='blue', edgecolor='black', bins=6)

# seaborn histogram
# sns.distplot(flights['arr_delay'], hist=True, kde=False,
#              bins=int(180/5), color = 'blue',
#              hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Technique 2')
plt.show()

from numpy import mean, std  # version >= 1.7.1 && <= 1.9.1
from math import sqrt
from statistics import stdev

# calculate length of 1st sample
n1 = len(technique1)

# calculate length of 2nd sample
n2 = len(technique2)

cohend = (mean(technique1) - mean(technique2)) / sqrt(
    ( ((n1-1) * (stdev(technique1) ** 2)) + ( (n2-1) * ( stdev(technique2) ** 2) )) / (n1 + n2-2))

print("Cohen's d = ",
      cohend)

from cliffs_delta import cliffs_delta
print("Cliff's delta: ", cliffs_delta(technique1, technique2))

from scipy.stats import bartlett

res = bartlett(technique1, technique2)
print(res)

#
res = mannwhitneyu(technique1, technique2, alternative='two-sided')
print(res)
# if p<0.05 significant difference


from scipy.stats import ttest_ind

res = ttest_ind(technique1, technique2, alternative="two-sided")
print(res)

from scipy.stats import ranksums

res = ranksums(technique1, technique2)
print(res)

from scipy.stats import kruskal
res = kruskal(technique1, technique2, technique1)
print(res)

import scikit_posthocs as sp
import numpy as np
#combine three groups into one array
data = np.array([technique1, technique2, [x+1 for x in technique1]])

#perform Nemenyi post-hoc test
res = sp.posthoc_nemenyi_friedman(data.T)
print(res)

#perform Dunn post-hoc test
res = sp.posthoc_dunn(data)
print(res)

#perform Wilcoxon post-hoc test
res = sp.posthoc_wilcoxon(data, p_adjust='fdr_bh')
print(res)
