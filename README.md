# pystuff

[![PyPI version](https://badge.fury.io/py/pystuff.svg)](https://badge.fury.io/py/pystuff)  [![GitHub version](https://badge.fury.io/gh/davidmnielsen%2Fpystuff.svg)](https://badge.fury.io/gh/davidmnielsen%2Fpystuff)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/davidmnielsen/pystuff/issues)

Some useful functions for data analysis in python.

- [Introduction](#introduction)
- [Handling time series](#handling-time-series) 
- [Periodogram](#periodogram)
- [Principal Component Analysis (PCA)](#principal-component-analysis)
- [Ensemble Subsampling](#ensemble-subsampling)
- [Statistics](#statistics)

## Introduction

#### Installation
```
pip install pystuff
```

#### Import pystuff
```python
import pystuff as ps
```

#### Import auxiliary packages
```python
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
```

#### Read example data
```python
# ERA-Interim SST anomalies, weighted by sqrt(cos(lat))
pathfile='/work/uo1075/u241292/data/ERAI/atmo/monavg_sst_w2_anom.nc'
ds=xr.open_dataset(pathfile)
lat=ds['lat'].values
lon=ds['lon'].values
myvar=ds['sst'].values
time=pd.to_datetime(ds['time'].values)
```

## Handling time series

```python
# Area Average
nasst, sasst_map = ps.getbox([0,60,280,360],lat,lon,myvar,returnmap=True)

# Standardize (center=True is default)
nasstn = ps.standardize(nasst)

# Running Mean
nasstn_rm = ps.runmean(nasstn,window=5*12)
nasstn_rm_fill = ps.runmean(nasstn,window=5*12,fillaround=True)

# Detrend (redundant with ddreg, to some extent)
# If returnTrend=False, only detended series is returned
nasstndt, slope, trend = ps.ddetrend(nasstn, returnTrend=True)

# The trend line can also be taken from:
trend2 = ps.ddreg(range(len(nasst)),nasst)

# Annual Mean
annual=ps.annualmean(nasstndt)

# Low-pass Lanczos Filter
dt=12 # month
cutoff=5 # years
low, low_nonan = ps.lanczos(nasstndt,dt*cutoff,returnNonan=True)

# Example Plot
fig = plt.figure(figsize=(9,4.5))

ax=fig.add_subplot(1,2,1)
ps.nospines(ax)
plt.axhline(0,color='grey',ls='--')
plt.plot(time,nasstn,'b',lw=0.2, label='Monthly NASST')
plt.plot(time,nasstn_rm,'b',lw=2, label='5-year running mean')
plt.plot(time,low+trend,'g',lw=2, label='5-year low-pass Lanczos filter')
plt.plot(time,trend,'-r', lw=2, label='Trend line')
plt.text(0.1,0.8,'Trend=%.2f std year$^{-1}$' %(float(slope)*12), color='r',
         transform=plt.gcf().transFigure)
plt.ylim((-2.5,2.5))
plt.ylabel('std. units')
plt.title('NASST')
plt.title('a', loc='left',fontweight='bold', fontsize=14)
ps.leg(loc='lower right', fontsize=9)

ps.usetex(False)
plt.tight_layout()
plt.show()

fig.savefig('/home/zmaw/u241292/scripts/python/pystuff/figs/timeseries.png')		
```

![alt text](https://raw.githubusercontent.com/davidmnielsen/pystuff/master/figs/timeseries.png "timeseries.png")

## Periodogram

```python
fig = plt.figure(figsize=(5,4.5))

ax=fig.add_subplot(111)
ps.nospines(ax)

# Calculate yearly periodogram on monthly data (Monte Carlo with 10k iterations)
f, psd, pctl, max5, meanRed = ps.periods(nasstndt, dt=12, nsim=10000)

plt.plot(f, psd,'b', lw=2, label='NASST Power Spectrum')
plt.plot(f,meanRed, 'r', lw=2, label='Mean of 10k Red-Noise Spectra')
plt.plot(f,pctl[:,0], '--k', label='80% Percentile')
plt.plot(f,pctl[:,1], '-k', label='90% Percentile')
plt.plot(f,pctl[:,2], '--g', label='95% Percentile')
plt.plot(f,pctl[:,3], '-g', label='99% Percentile')
plt.xlabel('Frequency [year$^{-1}$]')
plt.ylabel('PSD [Units$^{2}$ year]')
plt.xlim((0,2.5))
plt.title('Periodogram of NASST')
plt.title('b', loc='left',fontweight='bold', fontsize=14)
ps.leg(fontsize=10, frameon=True)

# Print maximum periods on graph
for i in range(5):
    t=plt.text(max5[i,0],max5[i,1],'%.0f yr' %round(max5[i,2]), color='blue')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='w'))


plt.tight_layout()
plt.show()

fig.savefig('/home/zmaw/u241292/scripts/python/pystuff/figs/periodogram.png')
```

![alt text](https://raw.githubusercontent.com/davidmnielsen/pystuff/master/figs/periodogram.png "periodogram.png")

## Principal Component Analysis (PCA)

```python
# Get data from any 3 grid cells
x1=myvar[:,100,100]
x2=myvar[:,100,200]
x3=myvar[:,100,300]
X=np.transpose([x1,x3,x2]) # np.shape(X) = (468, 3)

# Calculate PCA
scores, eigenvals, eigenvecs, expl, expl_acc, means, stds, north, loadings = ps.ddpca(X)

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
plt.text(0.6,46.5,'a',fontsize=14,fontweight='heavy')

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
plt.text(0.8,0.5,'Series 1',color='r')
plt.text(0.8,0.4,'Series 2',color='g')
plt.text(0.8,0.3,'Series 3',color='b')
plt.text(-0.4,0.9,'b',fontsize=14,fontweight='heavy')

ax1=f.add_subplot(gs[1, 1:-1])
ax1.set_xlim((-3,3)); plt.ylim((-3,3))
ax1.set(xlabel='PC1', ylabel='PC2')
ax1.axvline(0,color='k')
ax1.axhline(0,color='k')
ax1.plot(scores[:,0],scores[:,1],'o',markersize=3,
         markeredgecolor='lightgrey',markerfacecolor='lightgrey')
ax1.text(-2.8,2.5,'c',fontsize=15,fontweight='heavy')
ax2 = ps.twinboth(ax1)
ax2.set_xlim((-1.5,1.5)); plt.ylim((-1.5,1.5))
ax2.set_xlabel('PC1 Loadings', labelpad=3)
ax2.set_ylabel('PC2 Loadings', labelpad=14)
ax2.arrow(0,0,loadings[0,0],loadings[0,1],width=0.005,color='r',lw=2)
ax2.arrow(0,0,loadings[1,0],loadings[1,1],width=0.005,color='g',lw=2)
ax2.arrow(0,0,loadings[2,0],loadings[2,1],width=0.005,color='b',lw=2)

plt.tight_layout()
plt.show()

f.savefig('figs/pca.png')
```
![alt text](https://raw.githubusercontent.com/davidmnielsen/pystuff/master/figs/pca.png "pca.png")

## Ensemble Subsampling

```python
# Create a fake ensemble of nsim members of red noise (preserving the lag-dt) correlation
nsim=1000
dt=1
ens=ps.rednoise(len(nasstndt),ps.rhoAlt(nasstndt, dt=dt), nsim=nsim)

# Get Percentiles of Ensemble Spread
spread=ensPctl(ens,pctl=[0.025,0.975])

# Get the best 5% ensemble members, based on correlation
bestens, ensmean, nmembers = bestEns(ens,nasstndt,pctl=0.95)


# Plot
f=plt.figure(figsize=(8,6))

ax=f.add_subplot(2,1,1)
ps.nospines(ax)
plt.plot(time, ens,'lightgrey', lw=0.5)
# plt.fill_between(time, spread[:,0],spread[:,1],facecolor='k',alpha=0.1,label='2 std')
plt.plot(time, ens[:,1],'lightgrey', lw=1, label='All 10k Ensemble Members')
plt.plot(time, ens[:,1],'r', lw=0.5, label='Member 1')
plt.plot(time, np.nanmean(ens,axis=1),'k', lw=2, label='Full Ensemble Mean')
plt.plot(time, nasstndt,'b',lw=1,label='NASST')
plt.ylim((-6,6))
plt.ylabel('s.d.')
plt.title('Full Ensemble (10k Members)')
ps.leg(loc='upper right', frameon=True)

ax=f.add_subplot(2,1,2)
ps.nospines(ax)
plt.plot(time,bestens[:,0],'lightgrey', label='Best 5\% of Ensemble (50 members)')
plt.plot(time,bestens,'lightgrey')
plt.plot(time,ensmean,'k', lw=2, label='Best Ensemble Mean')
plt.plot(time,nasstndt,'b',lw=1,label='NASST')
plt.plot(time,low,'g',lw=2,label='NASST low-pass filt.')
plt.ylim((-6,6))
plt.ylabel('s.d.')
plt.title('Best Ensemble (50 Members)')
ps.leg(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
f.savefig('figs/ensemeble.png')
```
![alt text](https://raw.githubusercontent.com/davidmnielsen/pystuff/master/figs/ensemeble.png "ensemeble.png")

## Statistics

```python
# Annual means of monthly values
y=ps.annualmean(x)

# Autocorrelation of lag 1
r=ps.rho(x)

# Autocorrelation of lag n
r=ps.rhoAlt(x,dt=n)

# Rednoise time series
red=ps.rednoise(length, lag-1 correlation, nsim)

# Value of percentile
x95=getPercentile(x,pctl=0.95)

# Multiple Linear Regression (Uncertainties based on t-test)
X = [x1, x2, ... xn]
fitted, lo, up, r2, r2adj, [coefs] = ps.mlr(X, y, stds=2, returnCoefs=False, printSummary=False)

# Multiple Linear Regression (Uncertainties baed on Bootstrap)
fitted_0, up, lo, r2adj_0, r2adj_un, [r2_0, r2_unc, coefs_0, coefs] = ps.bootsmlr(X, y, n=1000, conflev=0.95, positions='new', uncertain='Betas', details=False, printSummary=False)

# Predict MLR, using coefs from the two above
pred, lo, up = ps.mlr_predict(X,coefs,stds=2)

# Residuals frmo MLR
res = ps.mlr_res(X, y)

# Histogram
binlims, x_count = ps.ddhist(x,vmin,vmax,binsize, zeronan=True, density=False)

# Order (sort ascending) x and y based on values of x
xo, yo = ps.order(x,y)

# Horizontal divergence
div = ps.hdivg(u,v,lat,lon,regulargrid=True)

# Vorticity (vertical component of relative vorticity)
vor = ps.hcurl(u,v,lat,lon,regulargrid=True)
```
## Miscellaneous

```python
# Saving CDO- and NCView-compatible NetCDF

# import pandas as pd
# mytime = pd.date_range('1979-01-01', periods=numYears, freq='1A')

newds = xr.Dataset(data_vars={'lat': (["lat"], lat),
                              'lon': (["lon"], lon),
                              'time': (["time"], np.arange(firstYear,lastYear+1)),
                              'clim' : (["lat","lon"], clim),
                              'trend': (["lat","lon"], trend),
                              'std'  : (["lat","lon"], std),
                              'stdDt': (["lat","lon"], stdDt),
                              'prop' : (["lat","lon"], prop),
                              'hs'   : (["time","lat","lon"], tempArray)})

newds['lat'].attrs  = {'standard_name':'lat','units':'degrees_north', '_CoordinateAxisType':'Lat'}
newds['lon'].attrs  = {'standard_name':'lon','units':'degrees_east',  '_CoordinateAxisType':'Lon'}
newds['time'].attrs = {'standard_name':'time', 'units':'years since 1979-01-01 00:00:00',
                       'calendar': 'proleptic_gregorian', '_CoordinateAxisType':'Time'}

newds['hs'].attrs    = {'standard_name':'hs',   'units':'m'}
newds['clim'].attrs  = {'standard_name':'clim', 'units':'m'}
newds['trend'].attrs = {'standard_name':'trend','units':'m/year'}
newds['std'].attrs   = {'standard_name':'std',  'units':'m'}
newds['stdDt'].attrs = {'standard_name':'stdDt','units':'m'}
newds['prop'].attrs  = {'standard_name':'prop', 'units':'%'}

newds.attrs = {'script' :'/home/u241292/scripts/python/acdtools/acdtools/Hs_Clim.ipynb',
               'content':'Statistics and Time Series of Significant Wave Height (Hs) from ERA5 remapcon2 WAM'}
newds.to_netcdf('/work/uo1075/u241292/outputs/CEModel/era5-rampcon2-wam_StatsTs_1979-2000_Hs.nc')
#                 unlimited_dims = 'time',
#                 encoding = {'time': {'dtype': 'datetime64[ns]'}})
```

