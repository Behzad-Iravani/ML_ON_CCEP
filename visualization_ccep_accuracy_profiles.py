# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:24:33 2023
visualziation the accuracy plot 
@author: behira
"""
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns


sns.set_theme(style="darkgrid")

# Define the significance level and degrees of freedom
significance_level = 0.975
degrees_of_freedom = 6

# Find the critical t value
critical_t_value = scipy.stats.t.ppf(significance_level, degrees_of_freedom-1)

window = 5
file = open('CCEP_time_resolved_scores.pkl', 'rb')
scores = pickle.load(file)


CV_score_time =  np.concatenate(scores, axis = 1)   
plt.plot(np.linspace(-200, 2000, CV_score_time.shape[0]) + window, np.mean(CV_score_time, axis=1),
         color= "black")
plt.fill_between(np.linspace(-200, 2000, CV_score_time.shape[0]) + window,
                          np.mean(CV_score_time, axis=1) - critical_t_value*np.std(CV_score_time, axis=1)/np.sqrt(6),
                         np.mean(CV_score_time, axis=1) + critical_t_value*np.std(CV_score_time, axis=1)/np.sqrt(6),
                          facecolor="blue", alpha=0.5)


plt.xlim([-100, 300])
plt.ylim([.20, .55])
plt.ylabel("SVM accuracy (95% CI)", fontsize=18)
plt.xlabel("Time (ms)", fontsize=18)

plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
# sns.despine(offset=10, trim=True);

plt.savefig("SVM ACC-6foldCV.png", dpi=300)
plt.savefig("SVM ACC-6foldCV.svg", format='svg')
plt.show()

