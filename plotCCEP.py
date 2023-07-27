# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:19:06 2023

@author: behira
"""

import CCEP_profile_ML as ccep
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def descriptive(data):
    out = np.mean(np.concatenate(tmp, axis = 0), axis = 0)
    return out/max(out)
def se_(data):
    # Define the significance level and degrees of freedom
    significance_level = 0.975
    out = np.concatenate(data, axis = 0)
    degrees_of_freedom = out.shape[0]
    se = np.std(out, axis = 0)/np.sqrt(degrees_of_freedom)
    # Find the critical t value
    critical_t_value = scipy.stats.t.ppf(significance_level, degrees_of_freedom-1)
    return critical_t_value*se

    
dat = ccep.data("F:\CCEP\script\ML_python_data.mat")

newdat = dat.epochs.get_data()
label = dat.epochs.events[:, -2]


fig, axes = plt.subplots(2,2)
times = np.linspace(-200,2000,newdat.shape[-1])
tmp = [newdat[ix,:,:] for ix in range(label.shape[0]) if label[ix] == 0]
axes[0,0].plot(times, descriptive(tmp), color = "black")
axes[0,0].set_xlim([-100,300])
axes[0,0].set_xlabel("Time (ms)")
axes[0,0].set_ylabel("Amplitude (95% CI)")
axes[0,0].set_title(dat.l.inverse_transform([0])[0])

axes[0,0].fill_between(times,
                         descriptive(tmp) - se_(tmp),
                         descriptive(tmp) + se_(tmp),
                          facecolor="blue", alpha=0.5)

tmp =[newdat[ix,:,:] for ix in range(label.shape[0]) if label[ix] == 1]
axes[0,1].plot(times, descriptive(tmp), color = "black")
axes[0,1].set_xlim([-100,300])
axes[0,1].set_title(dat.l.inverse_transform([1])[0])

axes[0,1].fill_between(times,
                         descriptive(tmp) - se_(tmp),
                         descriptive(tmp) + se_(tmp),
                          facecolor="blue", alpha=0.5)

tmp = [newdat[ix,:,:] for ix in range(label.shape[0]) if label[ix] == 2]
axes[1,0].plot(times, descriptive(tmp), color = "black")
axes[1,0].set_xlim([-100,300])
axes[1,0].set_title(dat.l.inverse_transform([2])[0])

axes[1,0].fill_between(times,
                         descriptive(tmp) - se_(tmp),
                         descriptive(tmp) + se_(tmp),
                          facecolor="blue", alpha=0.5)

tmp = [newdat[ix,:,:] for ix in range(label.shape[0]) if label[ix] == 3]
axes[1,1].plot(times, descriptive(tmp), color = "black")
axes[1,1].set_xlim([-100,300])
axes[1,1].set_title(dat.l.inverse_transform([3])[0])

axes[1,1].fill_between(times,
                         descriptive(tmp) - se_(tmp),
                         descriptive(tmp) + se_(tmp),
                          facecolor="blue", alpha=0.5)
plt.savefig("CCEP_profile.png", dpi=300)
plt.savefig("CCEP_profile.svg", format='svg')
plt.show()

