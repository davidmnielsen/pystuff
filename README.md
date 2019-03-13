# pystuff

Gathered useful functions to make daily life more simple.

This documentation is not complete.

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
#### North Atlantic SSTs

```python
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


