"""START OF DATA VISUALISATION"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Monovariate plots
num_var = df_full.select_dtypes(include=np.number) #numerical variables
cat_var = df_full.select_dtypes(include= 'category') #categorical variables

#Plotting distribution plots of numerical variables
sns.set_style('whitegrid')
sns.set_palette('cubehelix')

for col in num_var.columns:
    sns.distplot(num_var[col], axlabel=str(col), hist_kws={'color': 'darkblue'})
    mean = np.mean(num_var[col])
    median = np.median(num_var[col])
    plt.axvline(x=mean, c='orange', ls='--', lw = 2, label='mean')
    plt.axvline(x=median, c='g', ls='--', lw = 2, label='median')
    plt.legend()
    plt.title('Distribution plot of ' + col)
    plt.savefig('distplot of ' + col, dpi=400)
    plt.clf()

#Interestingly, most variables are normally distributed, with the exception of cost. BMI was more normally distributed than
#either height or weight individually. Age is not normally distributed, but I will leave it alone for now

#Plotting count plots of categorical variables
for col in cat_var.columns:
    sns.countplot(cat_var[col])
    plt.title('Count plot of ' + col)
    plt.savefig('Count plot of ' + col, dpi=400)
    plt.clf()

#Gender appears to be completely even. Same with pre-op medication 1. Most patients take pre-op medication 3,5 and 6
#Chinese and Singaporean dominated sample. Most patients show symptom 4

"""Bivariate plots of categorical plots and numerical plots against cost of care"""
#for categorical variables
sns.set_palette('deep')
sns.set_style('whitegrid')
for col in cat_var.columns:
    plt.figure(figsize=(9,6))
    sns.violinplot(num_var.amt_per_admission, cat_var[col], inner = 'box')
    plt.title('Violin plot of cost of cost per admission against ' + col)
    plt.savefig('violinplot_' + col, dpi=400)
    plt.tight_layout()
    plt.clf()

#Males seem to be slightly higher, with medical history 1 higher, medical history 5 is the one with super high cost
#med history 6 higher median, in general number of admissions doesn't say anything. Pre-op medication doesn't indicate much
#for race Chinese < Others < Malay < Indians, foreigners pay the most, while PR is next, Symptoms generally indicate
#slightly higher cost with the exception of symptom 5 which has substantially increased cost.

#for numerical variables
for col in num_var.columns.drop('amt_per_admission'):
    sns.regplot(num_var[col], num_var.amt_per_admission, scatter_kws={'alpha': 0.1})
    plt.title('Regression plot of cost per admission against ' + col)
    plt.savefig('regplot_' + col, dpi=400)
    plt.clf()

#BMI shows clearest trend of linear increase align with age. Interestingly, days_admitted does not contribute at all.
#All of the lab tests don't show anything. Weight shows more trend than height, so using bmi might dilute the effect of weight
#afterall

"""Faceted plots"""
sns.violinplot(num_var.amt_per_admission, cat_var.race, inner='box', hue = cat_var.lab_)
plt.title('Violin plot of cost of cost per admission against ')
plt.savefig('violinplot_', dpi=400)
plt.clf()

sns.stripplot(y = num_var.amt_per_admission, x = num_var.lab_result_1, hue = cat_var.race)

sns.violinplot(cat_var.race, num_var['bmi'], inner='box')

sns.countplot(cat_var.race, hue = cat_var.symptom_1)
sns.countplot(cat_var.race, hue = cat_var.resident_status)

#Symptom 5 appears a lot in Chinese, but chinese reported the least cost.
#Symptom 5 shows to have the most difference in cost across all race, as compared to the other symptoms

"""Correlation heatmap"""
plt.figure(figsize = (8,8))
sns.heatmap(num_var.corr(), xticklabels= True, linewidths=.5, cmap="YlGnBu")
plt.title('Correlation heatmap of all numerical variables')
plt.tight_layout()
plt.savefig("corr_heatmap",dpi = 400)

"""Move on to MACHINE LEARNING because the plots are not making sense"""

plt.figure(figsize=(9, 6))
sns.regplot(df_full.num_of_symptoms,df_full.amt_per_admission)
plt.title('Violin plot of cost of cost per admission against ' + col)
plt.savefig('violinplot_' + col, dpi=400)
plt.tight_layout()
plt.clf()
