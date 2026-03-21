import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind
import os

# 1. DATA INITIALIZATION
data = {
    "Language": ["English", "Spanish", "French", "German", "Italian", "Dutch", 
                 "Polish", "Portuguese", "Swedish", "Czech", "Hungarian", 
                 "Bulgarian", "Finnish", "Danish", "Slovak", "Latvian"],
    "Family": ["Germanic", "Romance", "Romance", "Germanic", "Romance", "Germanic", 
               "Slavic", "Romance", "Germanic", "Slavic", "Uralic", 
               "Slavic", "Uralic", "Germanic", "Slavic", "Baltic"],
    "Accuracy": [79.2, 73.1, 72.8, 71.5, 72.9, 70.4, 69.8, 71.2, 
                 69.5, 67.4, 66.8, 66.2, 66.1, 68.9, 66.5, 65.8],
    "Resource_Tier": ["Baseline", "HRL", "HRL", "HRL", "HRL", "HRL", 
                      "HRL", "HRL", "HRL", "MRL", "MRL", 
                      "MRL", "MRL", "MRL", "MRL", "MRL"]
}
df = pd.DataFrame(data)
os.makedirs("output", exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

# 2. INFERENTIAL STATS (T-Test & Cohen's d)
hrl = df[df['Resource_Tier'] == 'HRL']['Accuracy']
mrl = df[df['Resource_Tier'] == 'MRL']['Accuracy']
t_stat, p_val = ttest_ind(hrl, mrl, equal_var=False)

def cohens_d(x, y):
    pooled_std = np.sqrt((x.std()**2 + y.std()**2) / 2)
    return (x.mean() - y.mean()) / pooled_std
effect_size = cohens_d(hrl, mrl)

# 3. COMPUTATIONAL STATS: Bootstrapping (n=1000)
np.random.seed(42)
boot_diffs = []
for _ in range(1000):
    sample = df.sample(frac=1, replace=True)
    h_boot = sample[sample['Resource_Tier'] == 'HRL']['Accuracy']
    m_boot = sample[sample['Resource_Tier'] == 'MRL']['Accuracy']
    if not h_boot.empty and not m_boot.empty:
        boot_diffs.append(h_boot.mean() - m_boot.mean())
ci_95 = np.percentile(boot_diffs, [2.5, 97.5])

# 4. PREDICTIVE MODELING: OLS Regression
df_encoded = pd.get_dummies(df, columns=['Family'], drop_first=True)
X_cols = [col for col in df_encoded.columns if 'Family_' in col]
X = sm.add_constant(df_encoded[X_cols].astype(float))
y = df_encoded['Accuracy']
reg_model = sm.OLS(y, X).fit()

# Figure 1: Bar Chart
plt.figure(figsize=(12, 7))
colors = {"Baseline": "#2c3e50", "HRL": "#2980b9", "MRL": "#c0392b"}
ax = sns.barplot(x='Language', y='Accuracy', data=df.sort_values('Accuracy', ascending=False), 
                 hue='Resource_Tier', palette=colors, dodge=False)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=9, padding=3)
plt.title('Figure 1: Mean Accuracy across EU20 Language Tiers', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.ylim(60, 85)
plt.tight_layout()
plt.savefig("output/performance_bar_chart.png", dpi=300)

# Figure 2: Bootstrap Dist
plt.figure(figsize=(10, 6))
sns.histplot(boot_diffs, kde=True, color='teal', bins=30)
plt.axvline(ci_95[0], color='red', linestyle='--', label=f'95% CI Lower ({ci_95[0]:.2f})')
plt.axvline(ci_95[1], color='red', linestyle='--', label=f'95% CI Upper ({ci_95[1]:.2f})')
plt.title('Figure 2: Bootstrapped Stability of the Multilingual Tax', fontweight='bold')
plt.xlabel('Mean Difference (HRL - MRL) in Accuracy %')
plt.legend()
plt.tight_layout()
plt.savefig("output/bootstrap_distribution.png", dpi=300)

# Figure 3: Regression Plot
plt.figure(figsize=(10, 6))
plt.scatter(reg_model.fittedvalues, y, color='darkblue', alpha=0.6, s=100)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit')
plt.title('Figure 3: Regression Analysis - Model Predictive Accuracy', fontweight='bold')
plt.xlabel('Predicted Accuracy (%)')
plt.ylabel('Actual Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig("output/regression_diagnostic.png", dpi=300)

# --- THE PREFERRED OUTPUT ---
print("\n" + "="*80)
print("STATISTICAL EVIDENCE FOR CHAPTER 5")
print("="*80)
print(f"1. WELCH'S T-TEST: p-value = {p_val:.5f} | Cohen's d = {effect_size:.3f}")
print(f"2. BOOTSTRAPPED 95% CI (HRL - MRL difference): {ci_95[0]:.2f} to {ci_95[1]:.2f}")
print(f"3. REGRESSION FIT (R-Squared): {reg_model.rsquared:.4f}")

print("\n4. MEAN ACCURACY BY TIER:")
print(df.groupby('Resource_Tier')['Accuracy'].mean())

print("\n5. REGRESSION SUMMARY:")
print("===================================================================================")
# Prints the core coefficient table from the regression summary
print(reg_model.summary().tables[1])
print("===================================================================================")
print("="*80 + "\n")