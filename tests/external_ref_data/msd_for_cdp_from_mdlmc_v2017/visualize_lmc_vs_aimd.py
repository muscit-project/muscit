import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#path_aimd="/net/shared/dressler/siegert/intro_benchmark_lmc/csh2po4/aimd/aimd_orig_full_100_1000_full/msd_H_2.csv"
#path_lmc = "/net/shared/dressler/siegert/intro_benchmark_lmc/csh2po4/910ps-every100/partial/lmc2.out"
#path_aimd="/home/dressler/projects/il_mim_oac/sys/travis_ana_NtoO_v1/hou-neu/msd1/msd_H_#2.csv"
path_lmc = "lmc2.out"
#inter = 180
#replica = 100
#dt_lmc = 0.05

sweeps = 1000000
reset_freq = 10000
print_freq = 100
md_timestep_fs =  50


number_of_intervals = int(sweeps/reset_freq)
length_of_interval = int(reset_freq / print_freq)
#dt_lmc = md_timestep_fs/1000 * print_freq
dt_lmc = md_timestep_fs/1000 


raw_lmc_data = np.loadtxt(path_lmc, usecols=(0,1,2,3,4,5))
b = raw_lmc_data[:,2:5]
ac= np.reshape(b, (int(number_of_intervals),int(length_of_interval),3))
d = np.sum(ac, axis=2)
msd_mean = np.mean(d, axis=0)
#d = np.mean(c, axis=0)
#e = np.mean(d, axis=1)
#np.savetxt('msd_from_lmc.out', msd_mean)



#aimd_data = np.loadtxt(path_aimd, delimiter=';', skiprows=1)

#x_aimd = aimd_data[:,0]
#y_aimd = aimd_data[:,1]/10000

x_lmc = raw_lmc_data[:length_of_interval,0]*dt_lmc
y_lmc = msd_mean

np.savetxt('msd_from_lmc.out', np.column_stack((x_lmc, y_lmc)))

print(x_lmc)
#print(x_aimd)


#def diff_coef(t):
def diff_coef(t, D, n):
    return 6* D *t + n



print("diff coeff lmc in A**2/ps:")

len1 = x_lmc.shape[0]
fit_x_lmc = x_lmc[int(len1*0.2):int(len1*0.7)]
fit_y_lmc = y_lmc[int(len1*0.2):int(len1*0.7)]

#popt, pcov = curve_fit(func, xdata, ydata)
popt_lmc, pcov = curve_fit(diff_coef, fit_x_lmc, fit_y_lmc)
#popt, pcov = curve_fit(diff_coef, fit_x_lmc, fit_y_lmc, bounds=(0, [0.01, 100]))
print(popt_lmc[0])
#print(pcov)



#print("diff coeff aimd in A**2/ps:")

#len1 = x_aimd.shape[0]
#fit_x_aimd = x_aimd[int(len1*0.2):int(len1*0.7)]
#fit_y_aimd = y_aimd[int(len1*0.2):int(len1*0.7)]

#popt, pcov = curve_fit(func, xdata, ydata)
#popt_aimd, pcov = curve_fit(diff_coef, fit_x_aimd, fit_y_aimd)
#popt, pcov = curve_fit(diff_coef, fit_x_lmc, fit_y_lmc, bounds=(0, [0.01, 100]))
#print(popt_aimd[0])
#print(pcov)











fig, ax = plt.subplots()
#ax.plot(raw_lmc_datai[:inter,0]*dt_lmc, msd_mean)
#ax.plot(aimd_data[:,0], aimd_data[:,1]/10000)
ax.plot(x_lmc, y_lmc, label='cMD/LMC')
ax.plot(x_lmc, diff_coef(x_lmc, popt_lmc[0], popt_lmc[1]), label='fit cMD/LMC')
#ax.plot(x_aimd, y_aimd, label = 'AIMD')
#ax.plot(x_aimd, diff_coef(x_aimd, popt_aimd[0], popt_aimd[1]), label='fit AIMD')

ax.set(xlabel='time [ps]', ylabel='MSD  [$\AA^2$/ps]',
       title='Comparison of the MSDs from AIMD and cMD/LMC')

#ax.xlim(left=0)

#legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
legend = ax.legend(loc = 'best')
ax.grid()

fig.savefig("compare_msd_lmc_aimd.png")
#plt.show()

