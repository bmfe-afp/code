import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import config

# %%
mpl.rcParams['figure.autolayout'] = True

# %%
# log vix
vix_df = pd.read_excel('../data/cboe_vol.xlsx')
vix_df['Date'] = pd.to_datetime(vix_df['Date'], format='%d%b%Y')
vix_df.set_index('Date', inplace=True)
vix_s = vix_df['CBOE S&P100 Volatility Index - Close'].dropna()
vix_log_s = np.log(vix_s)

fig, axes = plt.subplots(3,1)
vix_log_s.plot(ax=axes[0])
plot_acf(vix_s, ax=axes[1])
plot_pacf(vix_s, ax=axes[2], lags=np.arange(0, 10))
fig.tight_layout()
fig.savefig('../plots/vix')

# %%
factors_df = pd.read_excel('../data/factors.xlsx', parse_dates=['date'], index_col='date')
factors_df.rename(columns={'Excess Return on the Market':'mkt', 'Small-Minus-Big Return':'smb',
                           'High-Minus-Low Return':'hml', 'Risk-Free Return Rate (One Month Treasury Bill Rate)':'rf', 'Momentum':'mom'}, inplace=True)

# %%
mod_hamilton = sm.tsa.MarkovAutoregression(vix_log_s, k_regimes=2, order=1, switching_ar=False, switching_variance=False)
res_hamilton = mod_hamilton.fit(method='bfgs', maxiter=100)
with open('../data/output.txt', 'a') as f:
    f.write("\n=== log(VIX) Markov Switching AR(1) ===\n")
    f.write(res_hamilton.summary().as_text())

# %%
fig, axes = plt.subplots(2,1, figsize=(18,8))
res_hamilton.smoothed_marginal_probabilities[0].plot(ax=axes[0])
res_hamilton.filtered_marginal_probabilities[0].plot(ax=axes[1])
plt.tight_layout()
fig.savefig('../plots/prob')

# %%
# select high vol and low vol data points
dates = res_hamilton.filtered_marginal_probabilities[0].index
high_vol_idx = dates[res_hamilton.filtered_marginal_probabilities[0]<.5]
low_vol_idx = dates[res_hamilton.filtered_marginal_probabilities[0]>=.5]
factors_high_vol = factors_df.loc[high_vol_idx,:]
factors_low_vol = factors_df.loc[low_vol_idx,:]

# explore dist, mean, cov, corr within regime
for k, v in {'low': factors_low_vol, 'high': factors_high_vol}.items():
    fig, ax = plt.subplots(1,1)
    v.hist(bins=100, ax=ax)
    file_name = k + '_vol_distribution'
    fig.savefig("../plots/{file_name}".format(file_name=file_name))
    mean = v.mean()
    covar = v.cov()
    corr = v.corr()
    with open(config.OUTPUT_FILE_PATH, 'a') as f:
        f.write('\n=== mean ' + k + ' vol ===\n')
    mean.to_csv(config.OUTPUT_FILE_PATH, mode='a')
    with open(config.OUTPUT_FILE_PATH, 'a') as f:
        f.write('\n=== covar ' + k + ' vol ===\n')
    covar.to_csv(config.OUTPUT_FILE_PATH, mode='a')
    with open(config.OUTPUT_FILE_PATH, 'a') as f:
        f.write('\n=== corr ' + k + ' vol ===\n')
    corr.to_csv(config.OUTPUT_FILE_PATH, mode='a')
    corr_heatmap_file_name = k + '_corr'
    fig, ax = plt.subplots(1,1)
    sns.heatmap(corr, ax=ax)
    fig.savefig("../plots/{file_name}".format(file_name=corr_heatmap_file_name))

# %%
# test if returns differ across regimes
# https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test
res_ttest = stats.ttest_ind(factors_high_vol, factors_low_vol, equal_var=True)
with open('../data/output.txt', 'a') as f:
    txt = "\n=== T-test (equal variance) for {columns} ===\n".format(columns=str(factors_df.columns.values))
    f.write(txt)
    f.write(str(res_ttest))

# %%
