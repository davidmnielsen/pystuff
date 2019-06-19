# pystuff

Simple functions for data analysis in Python. Below are some examples.

```python
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/zmaw/u241292/scripts/python/pystuff')
import pystuff.pystuff as ps
```

### Use Xarray and Pandas to read your data

```python
# ERA-Interim SST anomalies, weighted by sqrt(cos(lat))
pathfile='/work/uo1075/u241292/data/ERAI/atmo/monavg_sst_w2_anom.nc'
ds=xr.open_dataset(pathfile)
lat=ds['lat'].values
lon=ds['lon'].values
myvar=ds['sst'].values
time=pd.to_datetime(ds['time'].values)
```

### Some time-series handling

```python
# Area Average
nasst, sasst_map = ps.getbox([0,60,280,360],lat,lon,myvar,returnmap=True)

# Standardize (and center)
nasstn = ps.standardize(nasst)

# Running Mean
nasstn_rm = ps.runmean(nasstn,window=121)

# Detrend (redundant with ddreg, to some extent)
# If returnTrend=False, only detended series is returned
nasstndt, slope, trend = ps.ddetrend(nasstn, returnTrend=True)

# The trend can also be easily taken from:
# lfit = ps.ddreg(np.arange(len(nasst)),nasst)

# Low-pass Lanczos Filter
dt=12 # month
cutoff=5 # years
low, low_nonan = ps.lanczos(nasstndt,dt*cutoff,returnNonan=True)

# Plot
fig = plt.figure(figsize=(10,4))

fig.add_subplot(1,2,1)
plt.plot(time,nasstn,'b',lw=0.2)
plt.plot(time,nasstn_rm,'b',lw=2)
plt.plot(time,trend,'--r', lw=2)
plt.ylim((-2.5,2.5))
plt.ylabel('std. units')
plt.title('Detrended and smoothed with running mean')

fig.add_subplot(1,2,2)
plt.plot(time,nasstndt,'b',lw=0.2)
plt.plot(time,low,'b',lw=2)
plt.ylim((-2.5,2.5))
plt.title('Low-pass filtered')

plt.tight_layout()
plt.show()
```

![alt text](https://github.com/davidmnielsen/pystuff/blob/master/figs/timeseries.png "timeseries.png")

### Analysis on frequency domain

```python
fig = plt.figure(figsize=(12,5))

fig.add_subplot(1,2,1)
f, psd, conf, max5, psdm = ps.periods(low_nonan, 12)
plt.plot(f, psd, color='b', linewidth=2, label='Spectrum')
plt.plot(f,psdm, 'r', label='Mean spectrum of 1000 red-noise time series')
plt.plot(f,conf[50,:], '--r', label='50% conf. level')
plt.plot(f,conf[99,:], ':k', label='99% conf. level')
plt.plot(f,conf[95,:], '--k', label='95% conf. level')
plt.plot(f,conf[90,:], '-k', label='90% conf. level')
plt.xlabel('Frequency [year$^{-1}$]')
plt.ylabel('PSD [Units$^{2}$ year]')
for i in range(3):
    plt.text(max5[i,0],max5[i,1],'%.1f yr' %max5[i,2])
plt.xlim((0,2))
plt.legend(loc='upper right', fontsize='small', frameon=True)
plt.title('Spectrum of low-pass filtered data')

fig.add_subplot(1,2,2)
f, psd, conf, max5, psdm = ps.periods(nasstdt, 12)
plt.plot(f, psd, color='b', linewidth=2)
plt.plot(f,conf[50,:], '--r')
plt.plot(f,psdm, 'r')
plt.plot(f,conf[99,:], ':k')
plt.plot(f,conf[95,:], '--k')
plt.plot(f,conf[90,:], '-k')
plt.xlabel('Frequency [year$^{-1}$]')
plt.ylabel('PSD [Units$^{2}$ year]')
for i in range(3):
    plt.text(max5[i,0],max5[i,1],'%.1f yr' %max5[i,2])
#plt.xlim((0,2))
plt.title('Spectrum of original data')

plt.tight_layout()
plt.show()
```

![alt text](https://github.com/davidmnielsen/pystuff/blob/master/figs/periodogram.png "periodogram.png")

### PCA Example

```python
# Calculate PCA
scores, eigenvals, eigenvecs, expl, expl_acc, means, stds, north, loadings = ps.ddpca(X[rows,cols])

# Combo-plot
from matplotlib.gridspec import GridSpec
gs = GridSpec(nrows=2, ncols=4)
f=plt.figure(figsize=(6,6))

ax=f.add_subplot(gs[0, 0:2])
ps.usetex()
ps.nospines(ax)
plt.errorbar(np.arange(1,len(expl)+1),expl,yerr=[north,north], fmt='o',color='b',markeredgecolor='b')
for i in range(3):
    plt.text(i+1.2,expl[i],'$%.1f \pm %.1f$' %(expl[i],north[i]))
plt.xticks(np.arange(1,len(expl)+1,1))
plt.xlim((0.5,4))
plt.xlabel('PC')
plt.ylabel('Explained Variance [\%]')
plt.text(0.6,76,'a',fontsize=14,fontweight='heavy')

ax=f.add_subplot(gs[0, 2:])
ps.nospines(ax)
wdt=0.15
plt.axhline(0,color='k')
plt.bar(np.arange(3)-wdt,eigenvecs[0,:],facecolor='r',edgecolor='r',width=wdt)
plt.bar(np.arange(3)    ,eigenvecs[1,:],facecolor='g',edgecolor='g',width=wdt)
plt.bar(np.arange(3)+wdt,eigenvecs[2,:],facecolor='b',edgecolor='b',width=wdt)
plt.ylabel('Eigenvector values')
plt.xticks(np.arange(3)+0.125,['$1$','$2$','$3$'],usetex=True)
plt.xlabel('PC')
plt.axvline(0.625,color='lightgrey',ls='--')
plt.axvline(1.625,color='lightgrey',ls='--')
plt.text(-0.3,0.5,'Bykovsky',color='r')
plt.text(-0.3,0.4,'Muostakh N',color='g')
plt.text(-0.3,0.3,'Muostakh NE',color='b')
plt.text(-0.4,0.7,'b',fontsize=14,fontweight='heavy')

ax1=f.add_subplot(gs[1, 1:-1])
ax1.set_xlim((-3,3)); plt.ylim((-3,3))
ax1.set(xlabel='PC1', ylabel='PC2')
ax1.axvline(0,color='k')
ax1.axhline(0,color='k')
ax1.plot(scores[:,0],scores[:,1],'o',markeredgecolor='grey',markerfacecolor='grey')
ax1.text(-2.8,2.5,'c',fontsize=15,fontweight='heavy')
ax2 = twinboth(ax1)
ax2.set_xlim((-1.5,1.5)); plt.ylim((-1.5,1.5))
ax2.set_xlabel('PC1 Loadings', labelpad=3)
ax2.set_ylabel('PC2 Loadings', labelpad=14)
ax2.arrow(0,0,loadings[0,0],loadings[0,1],width=0.005,color='r',lw=2)
ax2.arrow(0,0,loadings[1,0],loadings[1,1],width=0.005,color='g',lw=2)
ax2.arrow(0,0,loadings[2,0],loadings[2,1],width=0.005,color='b',lw=2)

plt.tight_layout()
plt.show()
f.savefig('/work/uo1075/u241292/figures/draft/north_eigenvecs_biplot.png', dpi=300)
f.savefig('/work/uo1075/u241292/figures/draft/north_eigenvecs_biplot.pdf')
```
![alt text](https://github.com/davidmnielsen/pystuff/blob/master/figs/pca.png "pca.png")
