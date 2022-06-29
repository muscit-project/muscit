# coding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit



sweeps = 100000
reset_freq = 10
print_freq = 1
md_timestep_fs =  1


number_of_intervals = int(sweeps/reset_freq)
length_of_interval = int(reset_freq / print_freq)
dt_lmc = md_timestep_fs/1000


raw_lmc_data = np.loadtxt("lmc.out", skiprows=37, usecols=(0,1,2,3,4,5))
b = raw_lmc_data[:,2:5]
ac= np.reshape(b, (int(number_of_intervals),int(length_of_interval),3))
d = np.sum(ac, axis=2)
msd_mean = np.mean(d, axis=0)
np.savetxt('msd_from_lmc.out', msd_mean)




x_lmc = raw_lmc_data[:length_of_interval,0]*dt_lmc
y_lmc = msd_mean

#print(x_lmc)
def diff_coef(t, D, n):
    return 6* D *t + n



print("diff coeff lmc in A**2/ps:")

len1 = x_lmc.shape[0]
fit_x_lmc = x_lmc[int(len1*0.2):int(len1*0.7)]
fit_y_lmc = y_lmc[int(len1*0.2):int(len1*0.7)]

popt_lmc, pcov = curve_fit(diff_coef, fit_x_lmc, fit_y_lmc)
print(popt_lmc[0])
