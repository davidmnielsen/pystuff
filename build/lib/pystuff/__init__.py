#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is intended to put some usefull functions together, (organize my life/code,) and allow for some flexibility (in opposition to using built-in funcions of other software).
Content:
    1) bootscorr: calculates linear correlation coeffitient and significance level given n (default 10k) Bootstrap iterations.
    2) ddpca: performs a Principal Component Analysis (PCA) of a given dataset (centers and standardizes data), using its correlation matrix.
    3) ddetrend: removes linear trend (automatically treats NaN's)
    4) getbox: calculats the area mean (does not weight). Returns the mean (a float) and the map with NaN's outside the area.
    5) runmean: calculates running mean of given window size on x, and returns a series with same lenght as x
    6) ddreg: returns linear trend (not slope) of x, with same length as x (useful for poltting)
    7) standardize: Standardize (and center) time series
    8) compress: Compresses a time series of length x into a time series of length x-1. Useful to treat leap years.
    9) season1d: Calculates seasonal means of given monthly time series (starting in Jan). Returns [time, season] array. Default is JAS for season=0 (i.e. out[:,0] for JAS means in all years).
    10) order: sort a time series y in ascending order, and return a reordered time series x (w.r.t the sorting of y)
    
Found a bug? Please let me know:
davidnielsen@id.uff.br
"""

def bootscorr(x, y, n=10000, conflev=0.95, positions='new',details=False, std=False):

    """
    IN:
        x: numpy array, time series
        y: numpy array, time series
        n: number of bootstrap iterations (default = 10k)
        conflev: 0.95 and 0.6872 are equivalent to 2 and 1 sigmas, respectively. 
        positions: if "new", random pairwise positions (with replacement) will be generated.
        Otherwise, supply with numpy array with positions (same length as x and y, n-dimension in the columns). 
        details: Boolean, controls the output (defauls=False)
    
    OUT:
        if details==False: 
            r0  = linear pearson correlation coefficient between x and y [-1,1]
            lev = minimum significance level for which r0 is different than zero [0,1]
        else:
            rinf, rsup = lower and upper correlation coefficients, corrsponding to the tails of the distribution
                         containing conflev*100 percent of all n coefficients in between. 
            sig  = Boolean. True/False if significant at given conflev.
            rand = All n arrays of np.shape(x) each, containing the random positions generated.
    
    USAGE: 
    r, s = bootscorr(x,y)
    
    Found a bug? Please let me know:
    davidnielsen@id.uff.br
        
    """

    import numpy as np
    length=np.shape(x)[0] # time (or length) must be in the first dimension of x
    if length!=np.shape(y)[0]:
        print('ERROR: X and Y must have the same length.')
        print('Given dimensions were:')
        print('np.shape(x)=%s' %str(np.shape(x)))
        print('np.shape(y)=%s' %str(np.shape(y)))
        return
    else:
        
        # 1) Check given parameters
        if type(positions)==str:
            if positions=='new':
                # Create random positions
                import random
                rand=np.zeros((length,n))
                for nn in range(n):
                    for i in range(length):
                        rand[i,nn]=random.randint(0,length-1) 
            else:
                print('ERROR: Invalid position argument.')
                print('Must be eiter "new" or a numpy-array with shape (len(x),n)')
                return
        else:
            if len(x)!=np.shape(positions)[0]:
                print('ERROR: X, Y and given positions[0] must have the same length.')
                print('Given dimensions were:')
                print('np.shape(x)=%s' %str(np.shape(x)))
                print('np.shape(positions)[0]=%s' %str(np.shape(positions[0])))
                return
            elif n>np.shape(positions)[1]:
                print('ERROR: n must be <= np.shape(positions)[1]')
                print('Given dimensions were:')
                print('np.shape(n)=%s' %str(np.shape(n)))
                print('np.shape(positions)[1]=%s' %str(np.shape(positions[1])))
                return
            else:
                given_n=np.shape(positions)[1]
                rand=positions

        # 2) Schufle data
        schufx=np.zeros((length,n))
        schufy=np.zeros((length,n))
        for nn in range(n):
            for ii in range(len(x)):
                schufx[ii,nn]=x[int(rand[ii,nn])]
                schufy[ii,nn]=y[int(rand[ii,nn])]

        # 3) Calculate correlations
        r0=np.corrcoef(x,y)[0,1]
        corr=np.zeros(n)
        for nn in range(n):
            corr[nn]=np.corrcoef(schufx[:,nn],schufy[:,nn])[0,1]

        # 4) Significance test for given p-value (=1-conflev)
        sort=sorted(corr)
        tail=(1-conflev)/2
        qinf=round(n*tail)
        qsup=round(n-qinf)
        rinf=sort[qinf]
        rsup=sort[qsup]
        if rinf>0 or rsup<0:
            sig=True
        else:
            sig=False
        
        # 5) Check all possible p-values within n to get minimum significance level (minsig)
        lev=np.nan
        minsig=np.nan
        tails=np.arange(0.001,0.9,0.0001) # confidence level from 99.9% to 10%, changing every 0.01%
        for i in range(len(tails)):
            if np.isnan(minsig):
                tail=tails[i]/2
                qinf=round(n*tail)
                qsup=round(n-qinf)
                rrinf=sort[int(qinf)]
                rrsup=sort[int(qsup-1)]
                
                if rrinf>0 or rrsup<0:
                    minsig=tail*2
                    lev=(1-minsig)
       
    if std:
        return r0, np.std(corr)
	 
    if details:
        return r0, lev, rinf, rsup, sig, rand
    else:
        return r0, lev

#####################

def ddpca(x):
    
    """
    This functions calculates PCA from the input matrix x.
    x has variables organized in columns, and observations in rows.
    As it is, the data are cenered and devided by their respective standard deviation.
    """
    
    import numpy as np
    import pandas as pd
    
    # Get dimensions
    nobs=np.shape(x)[0]
    nvars=np.shape(x)[1]
    
    # Center
    means=np.mean(x,axis=0)
    mydata=x-means
    
    # Standardize
    stds=np.std(mydata,axis=0)
    mydata2=mydata*(1/stds)
    
    # Correlation matrix
    corrmat=np.corrcoef(mydata2, rowvar=False)
    
    # Eigenstuff
    eigenvals, eigenvecs = np.linalg.eig(corrmat)
   
    # Loadings (=correlation between x and scores)
    loadings=eigenvecs*np.sqrt(eigenvals)
 
    # Get PC's
    scores = mydata2 @ eigenvecs
    
    # Explained variance
    expl=np.sort(eigenvals)[::-1]*100/sum(eigenvals)
    expl_acc=expl.copy()
    for i in np.arange(1,len(expl)):
        expl_acc[i]=expl[i]+expl_acc[i-1]
    
    # North's Rule of Thumb
    north=expl*(1+np.sqrt(2/nobs))-expl
    
    return scores, eigenvals, eigenvecs, expl, expl_acc, means, stds, north, loadings

##### 3) Remove Linear Trend #####

def ddetrend(var,xvar=321943587416321,returnTrend=False,center=False):
    
    '''
    THIS IS WRONG!!!!!!
    USE `ddreg` INSTEAD
    '''

    import numpy as np
    from scipy import stats

    # If x is not given, use sequence
    if type(xvar)==int and xvar==321943587416321:
        xvar=np.arange(0,len(var))
        #print(np.shape(xvar))
        #print(np.shape(var))

    # Check and fix any NaN's
    if np.isnan(var).any():
        myvar=np.array(var)
        myxvar=np.array(xvar)
        nanmask = np.isnan(var)
        var_clean = myvar[np.where(nanmask==False)[0]]
        xvar_clean = myxvar[np.where(nanmask==False)[0]]
    else:
        var_clean=var.copy()
        xvar_clean=xvar.copy()

    if var_clean.size == 0:
        print("Time series provided is all NaN's.")
        print("Nothing done. Returning the input as is.")
        if returnTrend:
            return var, np.nan, np.zeros(np.shape(var))
        else:
            return var

    # Make linear model, calculate and remove trend
    slope, intercept, _, _, _ = stats.linregress(xvar_clean,var_clean)
    trend=np.array(xvar_clean)*slope+intercept
    if center:
        var_dt=var_clean-trend
    else:
        var_dt=var_clean-trend+np.mean(var_clean)

    # Put NaN's back in their places
    if np.isnan(var).any():
        var_dt_clean=np.zeros(np.shape(myvar))
        trend_clean=np.zeros(np.shape(myvar))
        t=0
        for i in range(len(myvar)):
            if nanmask[i]==False:
                var_dt_clean[i]=var_dt[t]
                trend_clean[i]=trend[t]
                t=t+1
            else:
                var_dt_clean[i]=np.nan
                trend_clean[i]=np.nan
    else:
        var_dt_clean=var_dt.copy()
        trend_clean=trend.copy()
    
    if returnTrend:
        return var_dt_clean, slope, trend_clean
    else:
        return var_dt_clean

##### 4) Get Box

def getbox(coords,lat,lon,data,returnmap=False):
    
    # data must be [time, lat, lon] or [lat, lon]
    # coords must be [lati,latf,loni,lonf]
    
    import numpy as np

    lati=coords[0]
    latf=coords[1]
    loni=coords[2]
    lonf=coords[3]

    # You can make this smarter to accept other polygons than rectangles
    # with len(coords) or something...    
    boxlat=[lati,lati,latf,latf,lati]
    boxlon=[loni,lonf,lonf,loni,loni]

    mylat=lat[np.where((lat<=latf) & (lat>=lati))]
    mylon=lon[np.where((lon<=lonf) & (lon>=loni))]

    inbox=data.copy()
    if np.size(np.shape(inbox))==3:
        # data is [time, lat, lon]
        inbox[:,np.where((lat>=latf) | (lat<=lati))[0],:]=np.nan
        inbox[:,:,np.where((lon>=lonf) | (lon<=loni))[0]]=np.nan
        temp=np.reshape(inbox,(np.shape(inbox)[0],np.shape(inbox)[1]*np.shape(inbox)[2]))
        meanbox=np.nanmean(temp,axis=1)
    elif np.size(np.shape(inbox))==2:
        # data is [lat, lon]
        inbox[np.where((lat>=latf) | (lat<=lati))[0],:]=np.nan
        inbox[:,np.where((lon>=lonf) | (lon<=loni))[0]]=np.nan
        temp=np.reshape(inbox,(np.shape(inbox)[0]*np.shape(inbox)[1]))
        meanbox=np.nanmean(temp,axis=0)
    else:
        print('ERROR: Number of dimensions must be 2 or 3.')
        meanbox=np.zeros(np.shape(inbox))
        return meanbox
    
    if returnmap:
        return meanbox, inbox
    else:
        return meanbox

########## 5) Running mean

def runmean(x,window=3,fillaround=False,weights=False,nanignore=False):
    
    '''
    Calculates a running mean of a given series x, using a window
    of given length. The window must be odd, ideally. Otherwise, 
    an approximation will be made.
    
    Optionally, an array of weights can be give (of same length of windowm, 
    otherwise length of weights will be used.) Useful to apply Lanczos weights
    to low-pass filter a time series.
    
    The option to fillaround does not work well with weights. Should not be used together.
    '''
    
    import numpy as np
    
    # Check for even window
    if (window % 2)==0:
        print('Window given is even.')
        print('Using window=%.0f instead.' %(window-1))
        window=window-1
    
    # Check for a too small window
    if window<3:
        print('Window given is too small.')
        print('Using minimum window=3 instead.')
        window=3
        
    # Check for weights
    if type(weights) is not bool:
        
        # Check if given window and weights have equal lengths
        if len(weights)!=window:
            #print('Window and array of weights have different lenghts.')
            print('pystuff.runmean: Using window with len(weights)=%d' %len(weights))
            window=len(weights)
        
    # This option will apply increasing windows on borders
    # so that the len(outseries)=len(inseries)
    if fillaround:
        increasingWindows=np.arange(3,window+1,2)
        #print(increasingWindows)
        x_rm=np.zeros(np.shape(x))
        for w in range(len(increasingWindows)):
            halfwindow=int((increasingWindows[w]-1)/2)
            for t in range(len(x)):
                # Is t on the edges?
                if t>=halfwindow and t<(len(x)-halfwindow):
                    # Apply weights?
                    if type(weights) is not bool:                        
                        if t<len(weights):
                            # t is on the left edge
                            if nanignore:
                                x_rm[t]=np.nansum(x[t-halfwindow:t+halfwindow]*\
                                           weights[-2*halfwindow-1:-1],
                                           axis=0)/np.nansum(weights[-2*halfwindow-1:-1])
                            else:
                                x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow]*\
                                               weights[-2*halfwindow-1:-1],
                                               axis=0)/np.sum(weights[-2*halfwindow-1:-1])
                        elif t>=(len(x)-len(weights)):
                            # t is on the right edge
                            if nanignore:
                                x_rm[t]=np.nansum(x[t-halfwindow:t+halfwindow]*\
                                               weights[:2*halfwindow],
                                               axis=0)/np.nansum(weights[:2*halfwindow])
                            else:
                                x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow]*\
                                               weights[:2*halfwindow],
                                               axis=0)/np.sum(weights[:2*halfwindow])
                        else:
                            # t is not on the edges
                            midw=(len(weights)/2)+1
                            if nanignore:
                                x_rm[t]=np.nansum(x[t-halfwindow:t+halfwindow]*\
                                               weights[int(midw-halfwindow):int(midw+halfwindow)],axis=0)/\
                                               np.nansum(weights[int(midw-halfwindow):int(midw+halfwindow)])
                            else:
                                x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow]*\
                                               weights[int(midw-halfwindow):int(midw+halfwindow)],axis=0)/\
                                               np.sum(weights[int(midw-halfwindow):int(midw+halfwindow)])                       
                    else:
                        # t not on the edges without weight
                        if nanignore:
                            x_rm[t]=np.nanmean(x[t-halfwindow:t+halfwindow+1],axis=0)
                        else:
                            x_rm[t]=np.mean(x[t-halfwindow:t+halfwindow+1],axis=0)
                else:
                    if halfwindow==1:
                        x_rm[t]=x[t]
                    else:
                        x_rm[t]=x_rm[t]
    else:
        x_rm=np.zeros(np.shape(x))
        halfwindow=int((window-1)/2)
        for t in range(len(x)):
            #print(halfwindow)
            if t>=halfwindow and t<(len(x)-halfwindow):
                if type(weights) is bool:
                    if nanignore:
                        x_rm[t]=np.nanmean(x[t-halfwindow:t+halfwindow+1],axis=0)
                    else:
                        x_rm[t]=np.mean(x[t-halfwindow:t+halfwindow+1],axis=0)
                else:
                    if nanignore:
                        x_rm[t]=np.nansum(x[t-halfwindow:t+halfwindow+1]*weights,axis=0)/np.nansum(weights)
                    else:
                        x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow+1]*weights,axis=0)/np.sum(weights)
            else:
                x_rm[t]=np.nan                
    return x_rm

#################### 6) ddreg
    
def ddreg(x,y,returnStats=False, returnCoefs=False):
    import numpy as np
    from scipy import stats
    slope, inter, r, p, se = stats.linregress(x,y)
    trend=np.array(x)*slope+inter
    if returnStats:
        return slope, r, p    
    elif returnCoefs:
        return slope, inter, trend
    else:
        return trend

################## 7) Standardize

def standardize(x,center=True,detrend=False):
    import numpy as np
    xn = np.full(np.shape(x),np.nan)
    if center:
        xn = (x - np.nanmean(x))/np.nanstd(x)
    else:
        xn = x/np.nanstd(x)
    if detrend:
        out=ddetrend(xn)
    else:
        out=xn.copy()
    return out

################## 8) Compress
def compress(x):
    import numpy as np
    lenx=len(x)
    lenout=lenx-1
    e=float(1/(lenout))
    e_acc=0
    out=np.zeros((lenout,))
    for i in range(lenout):
        if i==0:
            out[i]=x[i]
        else:
            e_acc=e_acc+e
            out[i]=e_acc*(x[i+1]-x[i])+x[i]
    return out
    
    
################ 8) Seasonal Means

def season1d(x,start='JAS',vmin=False,vmax=False):    
    import numpy as np
    
    # Create array of years
    if ~vmin or ~vmax:
        print('ymin and ymax not were not given.')
        print('Will assume 1979-2017 (incl).')#
        vmin=1979
        vmax=2018
    years=np.zeros((len(np.arange(vmin,vmax))*12,))
    y=0
    for i in np.arange(vmin,vmax):
        oneyear=np.zeros((12,)); oneyear.fill(i)
        years[0+y:12+y]=oneyear
        y=y+12

    # Create array of monthly positions
    positions=np.zeros((vmax-vmin,12))
    ano=vmin
    for i in np.arange(0,vmax-vmin,1):
        positions[i,:]=np.where(years==ano)[0][:]
        ano=ano+1
    positions=positions.astype(int); # 1979 to 2017 (incl.) : size 39

    # Get seasonal positions
    seas_pos=np.zeros((vmax-vmin,3,12))  # 1980 to 2017 (inlc.) : size 38
    for i in range(vmax-vmin-1):
        seas_pos[i,:,0]  = positions[i  , 6], positions[i  , 7], positions[i  , 8] # jas
        seas_pos[i,:,1]  = positions[i  , 7], positions[i  , 8], positions[i  , 9] # aso
        seas_pos[i,:,2]  = positions[i  , 8], positions[i  , 9], positions[i  ,10] # son
        seas_pos[i,:,3]  = positions[i  , 9], positions[i  ,10], positions[i  ,11] # ond
        seas_pos[i,:,4]  = positions[i  ,10], positions[i  ,11], positions[i+1, 0] # ndj
        seas_pos[i,:,5]  = positions[i  ,11], positions[i+1, 0], positions[i+1, 1] # djf
        seas_pos[i,:,6]  = positions[i+1, 0], positions[i+1, 1], positions[i+1, 2] # jfm
        seas_pos[i,:,7]  = positions[i+1, 1], positions[i+1, 2], positions[i+1, 3] # fma
        seas_pos[i,:,8]  = positions[i+1, 2], positions[i+1, 3], positions[i+1, 4] # mam
        seas_pos[i,:,9]  = positions[i+1, 3], positions[i+1, 4], positions[i+1, 5] # amj
        seas_pos[i,:,10] = positions[i+1, 4], positions[i+1, 5], positions[i+1, 6] # mjj
        seas_pos[i,:,11] = positions[i+1, 5], positions[i+1, 6], positions[i+1, 7] # jja

    # Make annual (seasonal) means [this is correct]
    season_mean=np.zeros((vmax-vmin,12)) # [year, seas, lat, lon]
    print(vmax-vmin)
    for t in range((vmax-vmin)):
        for s in range(12):
            #season_mean[t,s,:,:] = np.mean(myvar[seas_pos[t,:,s].astype(int),:,:],
            #             axis=0) # 1980 to 2017 (inlc.) : size 38
            season_mean[t,s] = np.mean(x[seas_pos[t,:,s].astype(int)], axis=0)
            
    return season_mean
    
########## Annual Mean (of monthly values) 
    
def annualmean(x):
    import numpy as np
    annual=np.zeros((len(x)/12,))
    year=0
    month=1
    for t in range(len(x)):
        if month==13:
            month=1
            year=year+1
        annual[year]=annual[year]+x[t]
        month=month+1
    for y in range(len(annual)):
        annual[y]=annual[y]/12
    return annual
    
######################### SIGNAL PROCESSING ######################### 
    
def rho(datax):
    # Calculates the lag-1 Autocorrelation Coefficient.
    import numpy as np  
    nrho=len(datax)
    sommesup=0
    sommeinf=0
    moy=np.sum(datax)/nrho
    datam=datax-moy
    for i in np.arange(1,nrho):
        j=i-1
        sommesup=sommesup+(datam[i]*datam[j])
        sommeinf=sommeinf+(datam[j]**2)
    rho=sommesup/sommeinf
    return rho

def rhoAlt(datax,dt=1):
    # Calculates the lag-dt Autocorrelation Coefficient, given the dt.
    import numpy as np  
    r=np.corrcoef(datax[0:-dt-1],datax[dt:-1])
    return r[0,1]

def rednoise(lenx, rho, nsim=1000, dist='normal', returnWhite=False, 
             mean=0, std=1, lo=-1, hi=1):
    '''
    Creates nsim time series of rednoise of length=lenx, with lag-1 autocorrelation rho.
    For normally-distributed series, user can provide mean and std.
    For uniformely-distributed series, user can provide low and high bounds.
    '''
    import numpy as np
    srho=(1-(rho**2))**(0.5)
    red=np.zeros((lenx,nsim))
    white=np.zeros((lenx,nsim))
    for j in range(nsim):
        for i in range(lenx):
            if dist=='normal':
                white[i,j]=np.random.normal(loc=mean, scale=std) 
            elif dist=='uniform':
                white[i,j]=np.random.uniform(low=lo, high=hi)
    for j in range(nsim):
        for i in range(lenx):
            if i==0:
                red[i,j]=white[i,j]*srho
            else:
                red[i,j]=rho*red[i-1,j]+white[i,j]*srho
    if returnWhite:
        return red, white
    else:
        return red


# def theored(dt,rho,meanP,f):
#     import numpy as np
#     fnyq=1/(2*dt)
#     theo=np.zeros((len(meanP,)))
#     for i in range(len(meanP)):
#         theo[i]=(1-rho**2)/(1-(2*rho*np.cos(np.pi*f[i]/fnyq))+rho**2)
#     theoun=theo[0]
#     theo[0]=0
#     Art=np.sum(theo)/(len(meanP))
#     theo[0]=theoun
#     Ax=np.sum(meanP)/len(meanP)
#     theo=theo*(Ax/Art);
#     return theo

def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.
    """
    import numpy as np
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def lanczos(x,cutoff,windowlen=False, fillaround=False, returnNonan=False):
    # Applies running mean (Convolution) using low-pass Lanczos weights
    import numpy as np
    if type(windowlen) is bool:
        windowlen  = cutoff+1
    #print(windowlen)
    weights    = low_pass_weights(int(windowlen), 1/cutoff)
    xlow       = runmean(x,window=int(windowlen),weights=weights,fillaround=fillaround)
    if returnNonan:
        xlow_nonan = xlow[~np.isnan(xlow)]
        return xlow, xlow_nonan
    else:
        return xlow
    
def periods(x,dt,returnPeriods=True, nsim=1000):
    import numpy as np
    from scipy import signal
    from scipy.stats.distributions import chi2
    
    # Calculate periodogram of x
    f, psd = signal.periodogram(x,fs=dt,detrend='linear')
    per=1/f
    max5=np.zeros((5,3))
    psdc=psd.copy()

    # Get the 5 larges periods
    for i in range(5):
        max5[i,0]=f[psdc==max(psdc)]
        max5[i,1]=psdc[psdc==max(psdc)]
        max5[i,2]=per[psdc==max(psdc)]
        psdc[psdc==max(psdc)]=0

    # Calculate periodograms of the nsim red-noise series
    fn=np.zeros((len(f),nsim))
    Pn=np.zeros((len(f),nsim))
    red=rednoise(len(x),rhoAlt(x,dt=dt),dist='normal',nsim=nsim)
    for i in range(nsim):
    #         if np.remainder(i,100)==0:
    #             print('%.0f %%' %(i/10))
        fn[:,i], Pn[:,i] = signal.periodogram(red[:,i],fs=dt,detrend='linear')

    # Mean spectrum of nsim simulations
    meanRed=np.mean(Pn,axis=1)

    # Get Percentiles of Distribution of Red-Noise Spectra
    pctl=np.zeros((len(f),4)) # 0.8, 0.9, 0.95, 0.99
    for i in range(len(f)):
        pctl[i,0]=getPercentile(Pn[i,:],0.80)
        pctl[i,1]=getPercentile(Pn[i,:],0.90)
        pctl[i,2]=getPercentile(Pn[i,:],0.95)
        pctl[i,3]=getPercentile(Pn[i,:],0.99)
        
    return f, psd, pctl, max5, meanRed

def getPercentile(x,pctl=0.95):
    '''
    Returns the values for the given percentile of variable
    '''
    import numpy as np
    n=len(x)
    sort=sorted(x)
    quant=round(n*pctl)
    qmean=round(n*0.5)
    rquant=sort[quant]
    rqmean=sort[qmean]
#     return abs(rqmean-rquant)
    return rquant 

def ensPctl(x,pctl=[0.025,0.975]):
    '''
    x is an ensemble f variables, such as the output of ps.rednoise()
    '''
    import numpy as np
    pctl=np.zeros((len(x),len(pctl)))
    for i in range(len(x)):
        pctl[i,0]=getPercentile(x[i,:],0.975)
        pctl[i,1]=getPercentile(x[i,:],0.025)
    return pctl

def bestEns(ens,x,pctl=0.95):
    import numpy as np
    nsim=np.shape(ens)[1]
    corrs=np.zeros((nsim,))
    for m in range(nsim):
        r=np.corrcoef(x,ens[:,m])
        corrs[m]=r[0,1]

    sortr=np.sort(corrs)
    threshold=getPercentile(sortr,pctl=pctl)

    bestens=ens[:,0]; bestens.fill(np.nan)
    count=0
    for m in range(nsim):
        if corrs[m]>=threshold:
#             bestens=bestens+ens[:,m]
            bestens=np.vstack([bestens,ens[:,m]])
            count=count+1
#     ensmean=np.
    return np.transpose(bestens), np.nanmean(bestens,axis=0), count
    
################ MLR
def mlr(X,y,stds=2,returnCoefs=False, printSummary=False, crossOrigin=False):
    '''
    This function calculates a Multiple Linear Regression (MLR) for any number of covariates (nvars)
    and the condidence intervals for given number of standar deviations (stds), 
    based on the principle described in:
    https://stats.stackexchange.com/questions/85560/shape-of-confidence-interval-for-predicted-values-in-linear-regression
    (see the answer by user ouranos).
    '''
    import numpy as np
    import numpy.matlib as npm
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    # Check the number of covariates
    if np.size(np.shape(X))==1:
        nvars=1
    elif np.size(np.shape(X))>1:
        X=np.transpose(np.stack(X))
        nvars=np.shape(X)[1]
    
    # Fit model
    if crossOrigin:
        xx = X.copy()
        ncoefs = nvars
    else:
        xx = sm.add_constant(X, prepend=True)
        ncoefs = nvars+1
    model  = smf.OLS(y,xx).fit()
    fitted = model._results.fittedvalues
    r2     = model._results.rsquared
    r2adj  = model._results.rsquared_adj    
    if printSummary:
        print(model.summary())

    # Store coefs
    coefs=np.zeros((ncoefs,2))
    for i in range(ncoefs):
        coefs[i,0]=model.params[i] 
        coefs[i,1]=model.bse[i]
       
    # Set up all possible combination of signs
    signs = np.zeros((2**(ncoefs),ncoefs))
    for j in range(ncoefs):
        count=((2**(ncoefs))/2**(j+1))
        fir=np.zeros((int(count),)); fir.fill(1)
        sec=np.zeros((int(count),)); sec.fill(-1)
        signs[:,j]=npm.repmat(np.hstack([fir,sec]),1,2**(j)) 
    
    # Make all possible lines, combining the signs of uncertainties
    lines=np.zeros((len(y),2**(ncoefs)))
    for i in range(2**(ncoefs)): ############# THE FOLLOWING IS STILL INCORRECT
        if nvars==1:
            if ~crossOrigin:
                lines[:,i] = coefs[0,0]+signs[i,0]*stds*coefs[0,1] + X*(coefs[1,0]+signs[i,1]*stds*coefs[1,1])
            else:
                lines[:,i] = X*(coefs[0,0]+signs[i,1]*stds*coefs[0,1])
        else:
            if ~crossOrigin:
                linterm = coefs[0,0]+signs[i,0]*stds*coefs[0,1]
                firstAngCoefPos = 1 
            else:
                linterm = 0
                firstAngCoefPos = 0
            angterm = 0
            for n in np.arange(firstAngCoefPos,ncoefs):
                angterm = angterm + X[:,n-1]*(coefs[n,0]+signs[i,n]*stds*coefs[n,1])
            lines[:,i] = linterm + angterm         
    
    combs=np.stack(lines,axis=0)
    lo=np.min(combs, axis=1)
    up=np.max(combs, axis=1)
    
    '''  
    ### This was very dumb: 
    if nvars==1:
        # Calculate lower and upper boudaries for uncertainties
        x1 = X                 
        comb1=x1*(coefs[1,0]-stds*coefs[1,1]) + coefs[0,0]-stds*coefs[0,1]
        comb2=x1*(coefs[1,0]+stds*coefs[1,1]) + coefs[0,0]-stds*coefs[0,1]
        comb3=x1*(coefs[1,0]-stds*coefs[1,1]) + coefs[0,0]+stds*coefs[0,1]
        comb4=x1*(coefs[1,0]+stds*coefs[1,1]) + coefs[0,0]+stds*coefs[0,1]
        combs=np.stack([comb1,comb2,comb3,comb4],axis=1)
        lo=np.min(combs, axis=1)
        up=np.max(combs, axis=1)
    
    elif nvars==2:
        # Calculate lower and upper boudaries for uncertainties
        x1 = X[:,0]
        x2 = X[:,1]
        comb1 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + coefs[0,0]-stds*coefs[0,1]
        comb2 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + coefs[0,0]+stds*coefs[0,1]
        comb3 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + coefs[0,0]+stds*coefs[0,1]
        comb4 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + coefs[0,0]-stds*coefs[0,1]
        comb5 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + coefs[0,0]-stds*coefs[0,1]
        comb6 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + coefs[0,0]+stds*coefs[0,1]
        comb7 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + coefs[0,0]-stds*coefs[0,1]
        comb8 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + coefs[0,0]+stds*coefs[0,1]
        combs=np.stack([comb1,comb2,comb3,comb4,comb5,comb6,comb7,comb8],axis=1)
        lo=np.min(combs, axis=1)
        up=np.max(combs, axis=1)
        
    elif nvars==3:
        x1 = X[:,0]
        x2 = X[:,1]
        x3 = X[:,2]
        comb1  = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb2  = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb3  = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb4  = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb5  = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb6  = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb7  = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb8  = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]-stds*coefs[0,1]
        comb9  = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb10 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb11 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb12 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]-stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb13 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb14 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]-stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb15 = x1*(coefs[1,0]-stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        comb16 = x1*(coefs[1,0]+stds*coefs[1,1]) + x2*(coefs[2,0]+stds*coefs[2,1]) + x3*(coefs[3,0]+stds*coefs[3,1]) + coefs[0,0]+stds*coefs[0,1]
        combs=np.stack([comb1,comb2, comb3, comb4, comb5, comb6, comb7, comb8,
                        comb9,comb10,comb11,comb12,comb13,comb14,comb15,comb16],axis=1)
        lo=np.min(combs, axis=1)
        up=np.max(combs, axis=1)
   '''  
    
    if returnCoefs:
        return fitted, lo, up, r2, r2adj, coefs
    else:
        return fitted, lo, up, r2, r2adj

def mlr_predict(X,coefs,stds=2):
    # Uses the same idea from ps.mlr to predict
    # Input:  X (the same provided to ps.mlr)
    #         coefs (the output from ps.mlr)
    # Usage:
    #         predicted, lo, up = ps.mlr_predict(X,coefs)
    import numpy as np
    import numpy.matlib as npm
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    # Check the number of covariates
    if np.size(np.shape(X))==1:
        nvars=1
    elif np.size(np.shape(X))>1:
        X=np.transpose(np.stack(X))
        nvars=np.shape(X)[1]
        
    # Set up all possible combination of signs
    signs = np.zeros((2**(nvars+1),nvars+1))
    for j in range(nvars+1):
        count=((2**(nvars+1))/2**(j+1))
        fir=np.zeros((int(count),)); fir.fill(1)
        sec=np.zeros((int(count),)); sec.fill(-1)
        signs[:,j]=npm.repmat(np.hstack([fir,sec]),1,2**(j))
        
    # Calculates predicted values
    if nvars==1:
        pred = coefs[0,0] + X*(coefs[1,0])
    else:
        linterm = coefs[0,0]
        angterm = 0
        for n in np.arange(1,nvars+1):
            angterm = angterm + X[:,n-1]*(coefs[n,0])
        pred = linterm + angterm       
        
    # Make all possible lines, combining the signs of uncertainties
    lines=np.zeros((len(X),2**(nvars+1)))
    for i in range(2**(nvars+1)):
        if nvars==1:
            lines[:,i] = np.squeeze(coefs[0,0]+signs[i,0]*stds*coefs[0,1] + X*(coefs[1,0]+signs[i,1]*stds*coefs[1,1]))
        else:
            linterm = coefs[0,0]+signs[i,0]*stds*coefs[0,1]
            angterm = 0
            for n in np.arange(1,nvars+1):
                angterm = angterm + X[:,n-1]*(coefs[n,0]+signs[i,n]*stds*coefs[n,1])
            lines[:,i] = linterm + angterm         
    combs=np.stack(lines,axis=0)
    lo=np.min(combs, axis=1)
    up=np.max(combs, axis=1)
    
    return pred, lo, up

def loo_predict(X, coefs):
    import numpy as np
    nvars = np.shape(coefs)[0]-1
    out = coefs[0,0] # the first coef is the intercept
    for i in range(nvars):
        out = out + X[i]*coefs[i+1,0]
    return out
    
def mlr_res(X,y):
    '''
    Similar to mlr, but returns only the residuals
    '''
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if np.size(np.shape(X))>1:
        X=np.transpose(np.stack(X))
    xx = sm.add_constant(X, prepend=True)
    model = smf.OLS(y,xx).fit()
    fitted = model._results.fittedvalues    
    res = y - fitted
    return res
    
################ Small Useful Stuff
    
def nospines(ax):
    import matplotlib.pyplot as plt
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
def leg(loc='best', fontsize='small', frameon=False):
    import matplotlib.pyplot as plt
    plt.legend(loc=loc, frameon=frameon, fontsize=fontsize)

def usetex(param=True):
    from matplotlib import rc
    rc('text', usetex=param)

def ddhist(x,vmin,vmax,binsize, zeronan=True, density=False):
    import numpy as np
    binlims=np.arange(vmin,vmax+binsize,binsize)
    x_count=np.zeros((len(binlims)-1,))
    for i in range(len(binlims)-1):
        x_count[i]=len(np.where(((x>binlims[i] )& (x<=binlims[i+1])))[0])
    if zeronan:
        x_count[x_count==0]=np.nan
    if density:
        x_count=x_count/np.nansum(x_count)
    return binlims, x_count

################## order

def order(base,target, returnIndices=False):
    import numpy as np
    baseline=base.copy()
    posi=np.argsort(baseline)
    yout=target[posi]
    '''
    for i in range(len(baseline)):
        mini = np.min(baseline)
        minipos = np.where(baseline==mini)[0]
        if len(minipos)>1:
            minipos = np.min(minipos)
        posi[i]=int(minipos)
        yout[i]=target[minipos]
        baseline[minipos]=10**10
    '''
    if returnIndices:
        return yout, posi
    else:
        return yout
 
##################### Load Coastal Erosion Rates

def loaderos(i,th=0.8,standardize=True):
    
    ### USAGE:
    # for in range(4):
    #     pos, pos_hi, pos_lo, pos_hilo, ero, name, time_ero = loaderos(i, th=0.8, standardize=True)
    
    import numpy as np
    import xarray as xr
    import pandas as pd
    
    import warnings
    warnings.filterwarnings("ignore")
    
    if i==0:
        series='vasi_marre'
        file_ero='/work/uo1075/u241292/data/COAST/Vasiliev/%s_from1979_dt.nc' %series
        ds_ero=xr.open_dataset('%s' %file_ero)
        erosion=ds_ero['erosion'].values
        erosion[np.where(erosion==-9999)]=np.nan
        marre=erosion
        if standardize:
            marre=(erosion-np.nanmean(erosion))/np.nanstd(erosion,axis=0)
        pos_marre   = np.where(~np.isnan(marre[1:-1]))[0]
        pos_marre_hi   = np.where((~np.isnan(marre[1:-1]))   & ((marre[1:-1]-np.nanmean(marre[1:-1]))/np.nanstd(marre[1:-1])>th))[0]
        pos_marre_lo   = np.where((~np.isnan(marre[1:-1]))   & ((marre[1:-1]-np.nanmean(marre[1:-1]))/np.nanstd(marre[1:-1])<-th))[0]
        pos_marre_hilo   = np.where((~np.isnan(marre[1:-1]))   & (abs((marre[1:-1]-np.nanmean(marre[1:-1]))/np.nanstd(marre[1:-1]))>th))[0]
        marre_short=marre[1:-1]

        
    elif i==1:
        series='grig_bykovsky'
        file_ero='/work/uo1075/u241292/data/COAST/Grig_Razu/%s_from1979_dt.nc' %series
        ds_ero=xr.open_dataset('%s' %file_ero)
        erosion=ds_ero['erosion'].values
        erosion[np.where(erosion==-9999)]=np.nan
        bykov=erosion
        if standardize:
            bykov=(erosion-np.nanmean(erosion))/np.nanstd(erosion,axis=0)
        pos_bykov   = np.where(~np.isnan(bykov[1:-1]))[0] # from 1980 to 2017 (incl.)
        pos_bykov_hi   = np.where((~np.isnan(bykov[1:-1]))   & ((bykov[1:-1]-np.nanmean(bykov[1:-1]))/np.nanstd(bykov[1:-1])>th))[0]
        pos_bykov_lo   = np.where((~np.isnan(bykov[1:-1]))   & ((bykov[1:-1]-np.nanmean(bykov[1:-1]))/np.nanstd(bykov[1:-1])<-th))[0]
        pos_bykov_hilo   = np.where((~np.isnan(bykov[1:-1]))   & (abs((bykov[1:-1]-np.nanmean(bykov[1:-1]))/np.nanstd(bykov[1:-1]))>th))[0]
        bykov_short=bykov[1:-1] # 1980 to 2017 (incl.) :size 38

        
    elif i==2:
        series='grig_muostakh_N'
        file_ero='/work/uo1075/u241292/data/COAST/Grig_Razu/%s_from1979_dt.nc' %series
        ds_ero=xr.open_dataset('%s' %file_ero)
        erosion=ds_ero['erosion'].values
        erosion[np.where(erosion==-9999)]=np.nan
        muostN=erosion
        if standardize:
            muostN=(erosion-np.nanmean(erosion))/np.nanstd(erosion,axis=0)
        pos_muostN  = np.where(~np.isnan(muostN[1:-1]))[0]
        pos_muostN_hi  = np.where((~np.isnan(muostN[1:-1]))  & ((muostN[1:-1]-np.nanmean(muostN[1:-1]))/np.nanstd(muostN[1:-1])>th))[0]
        pos_muostN_lo  = np.where((~np.isnan(muostN[1:-1]))  & ((muostN[1:-1]-np.nanmean(muostN[1:-1]))/np.nanstd(muostN[1:-1])<-th))[0]
        pos_muostN_hilo  = np.where((~np.isnan(muostN[1:-1]))  & (abs((muostN[1:-1]-np.nanmean(muostN[1:-1]))/np.nanstd(muostN[1:-1]))>th))[0]
        muostN_short=muostN[1:-1]

    
    elif i==3:
        series='grig_muostakh_NE'
        file_ero='/work/uo1075/u241292/data/COAST/Grig_Razu/%s_from1979_dt.nc' %series
        ds_ero=xr.open_dataset('%s' %file_ero)
        erosion=ds_ero['erosion'].values
        erosion[np.where(erosion==-9999)]=np.nan
        muostNE=erosion
        if standardize:
            muostNE=(erosion-np.nanmean(erosion))/np.nanstd(erosion,axis=0)
        pos_muostNE = np.where(~np.isnan(muostNE[1:-1]))[0]
        pos_muostNE_hi = np.where((~np.isnan(muostNE[1:-1])) & ((muostNE[1:-1]-np.nanmean(muostNE[1:-1]))/np.nanstd(muostNE[1:-1])>th))[0]
        pos_muostNE_lo = np.where((~np.isnan(muostNE[1:-1])) & ((muostNE[1:-1]-np.nanmean(muostNE[1:-1]))/np.nanstd(muostNE[1:-1])<-th))[0]
        pos_muostNE_hilo = np.where((~np.isnan(muostNE[1:-1])) & (abs((muostNE[1:-1]-np.nanmean(muostNE[1:-1]))/np.nanstd(muostNE[1:-1]))>th))[0]
        muostNE_short=muostNE[1:-1]

    time_ero=ds_ero['time'][1:-1].values
    
    if i==0:
        pos=pos_marre
        pos_hi=pos_marre_hi
        pos_lo=pos_marre_lo
        pos_hilo=pos_marre_hilo
        ero=marre_short
        name='Marre Sale'
    if i==1:
        pos=pos_bykov
        pos_hi=pos_bykov_hi
        pos_lo=pos_bykov_lo
        pos_hilo=pos_bykov_hilo
        ero=bykov_short
        name='Bykovsky'
    if i==2:
        pos=pos_muostN
        pos_hi=pos_muostN_hi
        pos_lo=pos_muostN_lo
        pos_hilo=pos_muostN_hilo
        ero=muostN_short
        name='Muostakh N'
    if i==3:
        pos=pos_muostNE
        pos_hi=pos_muostNE_hi
        pos_lo=pos_muostNE_lo
        pos_hilo=pos_muostNE_hilo
        ero=muostNE_short
        name='Muostakh NE'
        
    return pos, pos_hi, pos_lo, pos_hilo, ero, name, time_ero
    
###################### Horizontal Divergence and Vorticity

def hdivg(u,v,lat,lon,regulargrid=True):
    import numpy as np
    
    # u and v must be 2D only [lat, lon]
    
    # Initialize empty variable
    div = np.zeros(np.shape(u))
    
    # Calculate resolution
    if regulargrid:
        xres=abs(lon[1]-lon[0])
        yres=abs(lat[1]-lat[0])
        dy=110000*yres
    
    # Calculate horizontal divergence
    for y in np.arange(1,len(lat)-1):
        if ~regulargrid:
            xres=abs(lon[y]-lon[y-1])
            yres=abs(lat[y]-lat[y-1])
            dy=110000*yres
        dx=abs(110000*np.cos(lat[y]*(np.pi/180))*xres)
        for x in np.arange(1,len(lon)-1):
                div[y,x] = (u[y,x+1]-u[y,x-1])/(2*dx) + (v[y+1,x]-v[y+1,x])/(2*dy)
                
    return div

def hcurl(u,v,lat,lon,regulargrid=True):
    import numpy as np
    
    # u and v must be 2D only [lat, lon]
    
    # Initialize empty variable
    vort = np.zeros(np.shape(u))
    
    # Calculate resolution
    if regulargrid:
        xres=abs(lon[1]-lon[0])
        yres=abs(lat[1]-lat[0])
        dy=110000*yres
    
    # Calculate vertical component of the curl
    for y in np.arange(1,len(lat)-1):
        if ~regulargrid:
            xres=abs(lon[y]-lon[y-1])
            yres=abs(lat[y]-lat[y-1])
            dy=110000*yres
        dx=abs(110000*np.cos(lat[y]*(np.pi/180))*xres)
        for x in np.arange(1,len(lon)-1):
                vort[y,x] = (v[y,x+1]-v[y,x-1])/(2*dx) + (u[y+1,x]-u[y+1,x])/(2*dy)
                
                
    return vort 


#################### Bootstrapping Multiple Linear Regression

def bootsmlr(X, y, n=1000, conflev=0.95, positions='new', uncertain='Betas', details=False, printSummary=False):

    """
    Bootstrapping Multiple Linear Regression (MLR)

    IN:
        X: Numpy array, or list of numpy arrays, containing the independent variables [x1, x2, ... xn]
        y: Numpy array, the dependent variable.
        n: Integer, number of iterations (optional, default n=1000)
        conflev: Confidence level for the 2-tail test (optional, default conflev=0.95)
        positions: 'new' generates new random sample positions for Bootstrap, 
                   or numpy array shape(y) with fixed positions (optional, default='new')
        uncertain: 'Betas' calculates the uncertainties of the fitted model based on the 
                   distribution of regression coefficients, 'Fitted' uses the uncertainties from
                   the distribution of fitted values themselves (optional, default='Betas')
        details: Boolean, True returns more outputs, see below (optional, default=False)
        printSummary: Boolean, prints out the summary of the MLR for the normal non-shuffled
                      case (optional, default printSummary=False)
        
    OUT:
    if details==False:
       fitted: Numpy array, shape(y), with the fitted values
       up: Numpy array, shape(y), with the fitted values *plus* the uncertainty
       lo: Numpy array, shape(y), with the fitted values *minus* the uncertainty
       r2adj: Adjusted coefficient of determination
       r2adj_unc: The uncertainty of the adjusted coefficient of determination

    USAGE:
    fitted, up, lo, r2adj, r2adj_unc = bootsmlr(X,y)

    """

    import numpy as np
    
    # Check the number of covariates
    if np.size(np.shape(X))==1:
        nvars=1
        length=np.shape(X)[0]
    elif np.size(np.shape(X))>1:
        nvars=np.shape(X)[0]
        length=np.shape(X)[1]
        X=np.stack(X)
        
    # Check consistency in dimensions
    if length!=len(y):
        print('ERROR: X and Y must have the same length.')
        print('Given dimensions were:')
        print('np.shape(x)=%s' %str(np.shape(X)))
        print('np.shape(y)=%s' %str(np.shape(y)))
        return
    else:
        
        # 0) Check given parameters
        if type(positions)==str:
            if positions=='new':
                # Create random positions
                import random
                rand=np.zeros((length,n))
                for nn in range(n):
                    for i in range(length):
                        rand[i,nn]=random.randint(0,length-1) 
            else:
                print('ERROR: Invalid position argument.')
                print('Must be eiter "new" or a numpy-array with shape (len(X),n)')
                return
        else:
            if len(x)!=np.shape(positions)[0]:
                print('ERROR: X, Y and given positions[0] must have the same length.')
                print('Given dimensions were:')
                print('np.shape(x)=%s' %str(np.shape(x)))
                print('np.shape(positions)[0]=%s' %str(np.shape(positions[0])))
                return
            elif n>np.shape(positions)[1]:
                print('ERROR: n must be <= np.shape(positions)[1]')
                print('Given dimensions were:')
                print('np.shape(n)=%s' %str(np.shape(n)))
                print('np.shape(positions)[1]=%s' %str(np.shape(positions[1])))
                return
            else:
                # Use given positions
                given_n=np.shape(positions)[1]
                rand=positions
                
        # 1) MLR on original order
        fitted_0, lo_0, up_0, r2_0, r2adj_0, coefs_0 = mlr(X,y,returnCoefs=True,
                                                              printSummary=printSummary)
        
        # 2) Shufle data
        if nvars==1:
            schufx=np.zeros((length ,n))
        else:
            schufx=np.zeros((length ,n , nvars))
        schufy=np.zeros((length ,n ))
        for nn in range(n):
            for ii in range(length):
                if nvars==1:
                    schufx[ii,nn]=X[int(rand[ii,nn])]
                else:
                    schufx[ii,nn,:]=X[:,int(rand[ii,nn])]
                schufy[ii,nn]=y[int(rand[ii,nn])]

        # MLR on Shuffled Data
        fitted = np.zeros((length,n))
        r2 = np.zeros((n,))
        r2adj = np.zeros((n,))
        coefs = np.zeros((nvars+1,2,n)) # [intercept + xn, se, boots iterations]
        for nn in range(n):
            if nvars==1:
                fitted[:,nn], _, _, r2[nn], r2adj[nn], coefs[:,:,nn] = mlr(np.transpose(schufx[:,nn]),
                                                                          schufy[:,nn],returnCoefs=True)
            else:
                fitted[:,nn], _, _, r2[nn], r2adj[nn], coefs[:,:,nn] = mlr(np.transpose(schufx[:,nn,:]),
                                                                              schufy[:,nn],returnCoefs=True)
        
        # 5) Get confidence level
        corr=np.sqrt(r2)
        lev = getConfLevel(corr,n)

        # 6) Get 1-sigma values from distributinos of regression coefficients
        for vv in range(nvars+1):
            # Uncertainties of coefs are stored with the original coefs_0
            coefs_0[vv,1] = getOneSigma(coefs[vv,0,:],n,conflev)
        r2_unc = getOneSigma(r2,n,conflev)
        r2adj_unc = getOneSigma(r2adj,n,conflev)
        
        if uncertain=='Betas':
            # Set up all possible combination of signs
            signs = np.zeros((2**(nvars+1),nvars+1))
            for j in range(nvars+1):
                count=((2**(nvars+1))/2**(j+1))
                fir=np.zeros((int(count),)); fir.fill(1)
                sec=np.zeros((int(count),)); sec.fill(-1)
                signs[:,j]=np.matlib.repmat(np.hstack([fir,sec]),1,2**(j)) 

            # Make all possible lines, combining the signs of uncertainties
            if nvars>1:
                XX=np.transpose(X)
            else:
                XX=X.copy()
            lines=np.zeros((len(y),2**(nvars+1)))
            for i in range(2**(nvars+1)):
                if nvars==1:
                    lines[:,i] = coefs_0[0,0]+signs[i,0]*coefs_0[0,1] +\
                                 XX*(coefs_0[1,0]+signs[i,1]*coefs_0[1,1])
                else:
                    linterm = coefs_0[0,0]+signs[i,0]*coefs_0[0,1]
                    angterm = 0
                    for nn in np.arange(1,nvars+1):
                        angterm = angterm + XX[:,nn-1]*(coefs_0[nn,0]+signs[i,nn]*coefs_0[nn,1])
                    lines[:,i] = linterm + angterm
            combs=np.stack(lines,axis=0)
            lo=np.min(combs, axis=1)
            up=np.max(combs, axis=1)
        elif uncertain=='Fitted':
            fitted_unc = np.zeros((length,))
            for tt in range(length):
                fitted_unc[tt] = getOneSigma(fitted[tt,:],n,conflev)
            lo=fitted_0-fitted_unc
            up=fitted_0+fitted_unc
        else:
            a=0
            print('ERROR: invalid option for uncertainty.')
            print('Must be eiter: uncertain="Fitted" or uncertain="Betas".')
            if details:
                return a, a, a, a, a, a, a, a, a
            else:
                return a, a, a, a, a
        
    if uncertain=='Betas':
        #print('Using distribution of Betas to calculate uncertainties.')
        if details:
            return fitted_0, up, lo, r2adj_0, r2adj_unc, r2_0, r2_unc, coefs_0, coefs
        else:
            return fitted_0, up, lo, r2adj_0, r2adj_unc
    elif uncertain=='Fitted':
        #print('Using distribution of Fitted values to calculate uncertainties.')
        if details:
            return fitted_0, up, lo, r2adj_0, r2adj_unc, r2_0, r2_unc, coefs_0, coefs
        else:
            return fitted_0, up, lo, r2adj_0, r2adj_unc
    
        
def getConfLevel(x,n):
    '''
    Returns the quantiles (rrinf,rrsup) when one of them crosses the 0 line,
    i.e. testing the null hypothesis that the quantity equals zero.
    '''
    import numpy as np
    sort=sorted(x)
    lev=np.nan
    minsig=np.nan
    tails=np.arange(0.001,0.999,0.001) 
    for i in range(len(tails)):
        if np.isnan(minsig):
            tail=tails[i]/2
            qinf=round(n*tail)
            qsup=round(n-qinf)
            rrinf=sort[int(qinf)]
            rrsup=sort[int(qsup-1)]
            if rrinf>0 or rrsup<0:
                minsig=tail*2
                lev=(1-minsig)
    return lev

def getOneSigma(x,n,conflev):
    '''
    Returns the values for mean+std and mean-std (rinf, rsup)
    '''
    import numpy as np
    sort=sorted(x)
    tail=(1-conflev)/2
    qinf=round(n*tail)
    qsup=round(n-qinf)
    qmean=round(n*0.5)
    rinf=sort[qinf]
    rsup=sort[qsup]
    rmean=sort[qmean]
    return abs(rmean-rinf)

def twinboth(ax):
    newax = ax.figure.add_subplot(ax.get_subplotspec(), frameon=False)
    newax.xaxis.set(label_position='top')
    newax.yaxis.set(label_position='right', offset_position='right')
    newax.yaxis.get_label().set_rotation(-90) # Optional...
    newax.yaxis.tick_right()
    newax.xaxis.tick_top()
    return newax

def circlemap(ax):
    '''
    This function returns the circle boudary for Arctic maps made with Cartopy.
    USAGE:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111,projection=ccrs.NorthPolarStereo())
        ps.circlemap(ax)
        ax.coastlines()
        plt.show()
    '''
    import numpy as np
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

###################### ACD Utilities #########################


'''
Functions to work with the ACD Database (Lantuit et al. 2012)
David M Nielsen, david.nielsen@uni-hamburg.de
'''

def ACD_centroids(file='/work/uo1075/u241292/data/COAST/Lantuit2012/ACD_database'):
    '''
    Calculates the centroids, or centers of mass, of each coastal segment.
    Returns arrays of longitudes and latitudes. 
    '''
    import numpy as np
    import shapefile
    import pygeoif as pg
    
    r = shapefile.Reader(file)
    
    g=[]
    for s in r.shapes():
        g.append(pg.geometry.as_shape(s)) 

    s=np.zeros(len(g))
    cxx=np.zeros(len(g))
    cyy=np.zeros(len(g))
    for i in range(len(g)):
        m = pg.MultiPolygon(g[i]);
        n=m.wkt;
        o=n.replace("MULTIPOLYGON(((","")
        p=o.replace(")))","")
        p2=p.replace("))","")
        p3=p2.replace("((","")
        p4=p3.replace(")","")
        p5=p4.replace("(","")
        q=p5.replace(", ",",")
        r=q.replace(" ",",")
        npoints=(r.count(",")+1)/2
        #print(npoints)
        s[i]=npoints
        t=r.split(",")

        u=np.zeros(len(t))
        for ii in range(len(t)):
            u[ii]=float(t[ii])

        x=u[np.arange(0,len(u),2)]
        y=u[np.arange(1,len(u)+1,2)]

        cxx[i]=np.mean(x)
        cyy[i]=np.mean(y)
    return cxx, cyy
    
def ACD_floatvars(file='/work/uo1075/u241292/data/COAST/Lantuit2012/ACD_database', returnNames=True):
    '''
    Returns the names and data for the float-type variables in the ACD.
    '''
    import shapefile
    import numpy as np
    sf = shapefile.Reader(file)
    nsegments=len(sf.shapeRecords())
    fields = sf.fields[1:] 
    nfields=len(fields)
    field_names = [field[0] for field in fields]
    field_types = [field[1] for field in fields]

    # Get Float Record Names
    float_record_names=[]
    for f in range(len(fields)):
        if field_types[f]=='F':
            float_record_names.append(field_names[f])  
            
    # Get Float Record Data
    for t in range(len(float_record_names)):
        print('%d: %s' %(t,float_record_names[t]))
        temp=np.zeros((len(sf.shapeRecords())))
        i=0
        for r in sf.shapeRecords():
            atr = dict(zip(field_names, r.record))
            temp[i]=atr[float_record_names[t]]
            i=i+1
        if t==0:
            float_record_data=temp.copy()
        else:
            float_record_data=np.vstack([float_record_data,temp])
    print(float_record_data.shape)
    if returnNames:
        return float_record_data, float_record_names
    else:
        return float_record_data
   
def ACD_segments():
    from cartopy.io.shapereader import Reader
    from cartopy.feature import ShapelyFeature
    import cartopy.crs as ccrs
    ACD_file='/work/uo1075/u241292/data/COAST/Lantuit2012/ACD_database'
    shapes = ShapelyFeature(Reader(ACD_file).geometries(),ccrs.PlateCarree())
    return shapes



def draw_text(ax, text='This is a piece of text.',size=10,frameon=True,loc='lower left'):
    from matplotlib.offsetbox import AnchoredText
    at = AnchoredText(text, loc=loc, prop=dict(size=size), frameon=frameon)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


################# Circle Averages
def circleavg_nsidc(plat,plon,r,xgrid,ygrid,lat,lon,data,returnMap=False, returnOnes=False):
    """
    This function returns time series of provided NSIDC data averaged over a circle 
    of radius r around the center located at plat,plon.
    r must me given in meters. 
    nsidc data dimensions can be [ygrid,xgrid] or [time,ygrid,xgrid]
    """
    import numpy as np
    
    # Number of dimensions (is it a time slice, or time series?)
    dims=np.size(data.shape)
    
    # Find the gridcell closest to the center (plat,plon)
    latdifs=abs(lat-plat)
    londifs=abs(lon-plon)
    sumdifs=latdifs+londifs
    mycoords=np.where(sumdifs==np.min(sumdifs))[:]
    mycoords0=mycoords[0][0]
    mycoords1=mycoords[1][0]-1

    # Define rectangle of size 2rx2r around the center
    minx=np.where(xgrid==(xgrid[mycoords0]-r))[0][0]
    maxx=np.where(xgrid==(xgrid[mycoords0]+r))[0][0]
    miny=np.where(ygrid==(ygrid[mycoords1]-r))[0][0]
    maxy=np.where(ygrid==(ygrid[mycoords1]+r))[0][0]
    rangex=np.arange(minx,maxx)
    rangey=np.arange(maxy,miny)
    
    # Prepare output
    mycircle=data.copy()
    mycircle.fill(np.nan)
    if returnOnes:
        myones=data.copy()
        myones.fill(np.nan)

    # Check distance of each grid cell from center
    for x in rangex:
        for y in rangey:
            dist_to_center=np.sqrt(abs(xgrid[mycoords0]-xgrid[x])**2 + abs(ygrid[mycoords1]-ygrid[y])**2)
            if dist_to_center<r:
                if dims==3:
                    mycircle[:,x,y]=data[:,x,y]
                    if returnOnes:
                        myones[:,x,y]=1
                elif dims==2:
                    mycircle[x,y]=data[x,y]
                    if returnOnes:
                        myones[x,y]=1
                    
    # Make time series
    if dims==3:
        mycircleavg=np.nanmean(np.nanmean(mycircle,axis=1),axis=1)
    if dims==2:
        mycircleavg=np.nanmean(np.nanmean(mycircle,axis=1),axis=0)
        
    # Return
    if returnMap:
        if returnOnes:
            return mycircleavg, mycircle, myones
        else:
            return mycircleavg, mycircle
    else:
        if returnOnes:
            return mycircleavg, myones
        else:
            return mycircleavg

def deg2m(r, clat, clon):
    '''
    Converts from degree to meters, at a certain latitude.
    '''
    import pyproj
    geod84 = pyproj.Geod(ellps='WGS84')
    _, _, dist_m = geod84.inv(clon, clat,  clon+r, clat)
    return dist_m

def m2deg(r, clat, clon):
    '''
    Converts from meters to degree (departing from) a certain latitude, towards North.
    '''
    import pyproj
    geod84 = pyproj.Geod(ellps='WGS84')
    _, _, out_z = geod84.fwd(clon, clat, 90, r, radians=False)
    return 90-abs(out_z)

def isleap(year):
    import calendar
    return calendar.isleap(year)

def scale_bar(ax, length=None, location=(0.7, 0.05), linewidth=3, color='y',fontsize=15):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
    """
    
    import cartopy.crs as ccrs
    import numpy as np
    import matplotlib.pyplot as plt

    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color=color, linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom', color=color, fontsize=fontsize)

def normBetween(x, objMin=0, objMax=365):
    import numpy as np
    xMin = np.nanmin(x)
    xMax = np.nanmax(x)
    out = np.zeros(np.shape(x))
    out.fill(np.nan)
    for i in range(len(x)):
        out[i] = ((x[i]-xMin)*(objMax-objMin))/(xMax-xMin) + objMin
    return out

def rmse(x,y):
    import numpy as np
    return np.sqrt(np.nansum((x-y)**2))

def bootSeries(x, y, n=1000, length=0):
    '''
    Takes as input two time series, x and y.
    Generated time series of given length, sampled with replacement from x and y.
    Default length is len(x), which should be equal to len(y).
    '''
    import numpy as np
    if length == 0:
        length = len(x)
    else:
        
        # Generate random positions
        import random
        rand=np.zeros((length,n))
        for nn in range(n):
            for i in range(length):
                rand[i,nn]=random.randint(0,len(x)-1) # must sample from entire length
        
        # Shuffle time series (actually, sample with replacement)             
        schufx = np.zeros((length,n))
        schufy = np.zeros((length,n))
        for nn in range(n):
            for ii in range(length):
                schufx[ii,nn] = x[int(rand[ii,nn])]
                schufy[ii,nn] = y[int(rand[ii,nn])]
        return schufx, schufy 

def extendSeries(x, xTime, longTime, allowNegatives=False, returnLongTrend=False):
    '''
    Extends time series x, which comprehends xTime, back in time to cover longTime.
    Generated time series respects the lag-1 correlation (red noise), mean, std and trend of the
    given time series x. 
    x must not have NaN's.
    Input:
        xTime = np.arange(1980,2020)
        longTime = np.arange(1950,2020)
        x = any time series with length = len(xTime)
    Output:
        xLong = a time series identical to x during xTime, and randomly generated
        mantaining its same statistical characteristics in the remaining years.
    '''
    import numpy as np
    
    redTime = np.arange(longTime[0],xTime[0])
    
    lenTimeX = len(xTime)
    lenLongTime = len(longTime)
    lenRedTime = len(redTime)
    
    xTrend  = np.zeros((lenLongTime)); xTrend.fill(np.nan)
    xDt     = np.zeros((lenLongTime)); xDt.fill(np.nan)
    #xRed    = np.zeros((lenRedTime)); xRed.fill(np.nan)
    xLong   = np.zeros((lenLongTime)); xLong.fill(np.nan)
    
    # Get the trend of x and extend it back during longTime
    a, b, t = ddreg(xTime, x, returnCoefs=True)
    xTrend = a*longTime + b
    if ~allowNegatives:
        xTrend[xTrend<0]=0
        
    # Detrend the real time series 
    start = lenLongTime-np.where(longTime==xTime[0])[0]
    xDt[-int(start):] = x - xTrend[-int(start):]  + np.mean(x)
    
    # Create red noise time series with std of original detrended series
    xRed = rednoise(lenRedTime, rho(xDt[~np.isnan(xDt)]), nsim=1,  mean=0,
               std = np.std(xDt[~np.isnan(xDt)]))
    
    # Give it the correct mean
    targetMean = np.mean(xDt[~np.isnan(xDt)])
    redMean = np.mean(xRed)
    if redMean>targetMean:
        xRed = xRed - abs(redMean - targetMean)
    elif redMean<targetMean:
        xRed = xRed + abs(redMean - targetMean)
        
    # Put them together
    fullDt = np.hstack((np.squeeze(xRed), xDt[~np.isnan(xDt)]))
    fullWt = fullDt + xTrend
    fullWt[xTrend==0] = fullWt[xTrend==0] - np.mean(fullWt[xTrend==0])
    fullWt[xTrend>0]  = fullWt[xTrend>0]  - np.mean(fullDt[xTrend!=0])
    if ~allowNegatives:
        fullWt[fullWt<0] = 0
   
    # Make sure the last bit to be identical (correct for minor differences)
    fullWt[-int(start):] = x 
     
    if returnLongTrend:
        return fullWt, xTrend
    else:
        return fullWt


def removeAfromB(A, B):
    import numpy as np
    trend = ddreg(A, B)
    return B -trend + np.mean(B)

def scoresFromPCA(X, eigenvecs, means=[0], stds=[0]):
    '''
    Returns the scores (PCA time series) for given data X and eigenvectors.
    Means and Stds of X are optional, in case X is the same for PCA, 
    but necessary if applying the transformation to other data.
    '''
    import numpy as np
    if len(means)==1:
        means = np.mean(X,axis=0)
    if len(stds)==1:
        stds = np.std(X,axis=0)
    pcs = np.zeros((X.shape))
    for p in range(X.shape[1]): # pc
        for v in range(X.shape[1]): # var
            pcs[:,p] = pcs[:,p] + ((X[:,v]-means[v])/stds[v])*eigenvecs[v,p]
    return pcs

def boots_r2(x, y, n=10000, pval=0.05, positions='new'):

    """
    Better version of bootscorr, for R**2.
    mu, std, lo, hi = boots_r2(x, y, n=10000, pval=0.05, positions='new')
    """

    import numpy as np
    length=np.shape(x)[0] # time (or length) must be in the first dimension of x
    if length!=np.shape(y)[0]:
        print('ERROR: X and Y must have the same length.')
        print('Given dimensions were:')
        print('np.shape(x)=%s' %str(np.shape(x)))
        print('np.shape(y)=%s' %str(np.shape(y)))
        return
    else:
        
        # 1) Check given parameters
        if type(positions)==str:
            if positions=='new':
                # Create random positions
                import random
                rand=np.zeros((length,n))
                for nn in range(n):
                    for i in range(length):
                        rand[i,nn]=random.randint(0,length-1) 
            else:
                print('ERROR: Invalid position argument.')
                print('Must be eiter "new" or a numpy-array with shape (len(x),n)')
                return
        else:
            if len(x)!=np.shape(positions)[0]:
                print('ERROR: X, Y and given positions[0] must have the same length.')
                print('Given dimensions were:')
                print('np.shape(x)=%s' %str(np.shape(x)))
                print('np.shape(positions)[0]=%s' %str(np.shape(positions[0])))
                return
            elif n>np.shape(positions)[1]:
                print('ERROR: n must be <= np.shape(positions)[1]')
                print('Given dimensions were:')
                print('np.shape(n)=%s' %str(np.shape(n)))
                print('np.shape(positions)[1]=%s' %str(np.shape(positions[1])))
                return
            else:
                given_n=np.shape(positions)[1]
                rand=positions

        # 2) Schufle data
        schufx=np.zeros((length,n))
        schufy=np.zeros((length,n))
        for nn in range(n):
            for ii in range(len(x)):
                schufx[ii,nn]=x[int(rand[ii,nn])]
                schufy[ii,nn]=y[int(rand[ii,nn])]

        # 3) Calculate correlations
        r0=np.corrcoef(x,y)[0,1]
        corr=np.zeros(n)
        for nn in range(n):
            corr[nn]=np.corrcoef(schufx[:,nn],schufy[:,nn])[0,1]**2

        # 4) Params
        mu = np.mean(corr)
        std = np.std(corr)
        
        # 5) Quantiles
        lo = np.sort(corr)[round(len(corr)*pval)]
        hi = np.sort(corr)[round(len(corr)*(1-pval))]
       
        return mu, std, lo, hi

def boots_r2_sklearn(x, y, n=10000, pval=0.05, positions='new'):

    """
    Using the sklearn version to allow for negative R**2 values.
    Better version of bootscorr, for R**2.
    mu, std, lo, hi = boots_r2_sklearn(x, y, n=10000, pval=0.05, positions='new')

    """

    import numpy as np
    from sklearn import metrics
    
    length=np.shape(x)[0] # time (or length) must be in the first dimension of x
    if length!=np.shape(y)[0]:
        print('ERROR: X and Y must have the same length.')
        print('Given dimensions were:')
        print('np.shape(x)=%s' %str(np.shape(x)))
        print('np.shape(y)=%s' %str(np.shape(y)))
        return
    else:
        
        # 1) Check given parameters
        if type(positions)==str:
            if positions=='new':
                # Create random positions
                import random
                rand=np.zeros((length,n))
                for nn in range(n):
                    for i in range(length):
                        rand[i,nn]=random.randint(0,length-1) 
            else:
                print('ERROR: Invalid position argument.')
                print('Must be eiter "new" or a numpy-array with shape (len(x),n)')
                return
        else:
            if len(x)!=np.shape(positions)[0]:
                print('ERROR: X, Y and given positions[0] must have the same length.')
                print('Given dimensions were:')
                print('np.shape(x)=%s' %str(np.shape(x)))
                print('np.shape(positions)[0]=%s' %str(np.shape(positions[0])))
                return
            elif n>np.shape(positions)[1]:
                print('ERROR: n must be <= np.shape(positions)[1]')
                print('Given dimensions were:')
                print('np.shape(n)=%s' %str(np.shape(n)))
                print('np.shape(positions)[1]=%s' %str(np.shape(positions[1])))
                return
            else:
                given_n=np.shape(positions)[1]
                rand=positions

        # 2) Schufle data
        schufx=np.zeros((length,n))
        schufy=np.zeros((length,n))
        for nn in range(n):
            for ii in range(len(x)):
                schufx[ii,nn]=x[int(rand[ii,nn])]
                schufy[ii,nn]=y[int(rand[ii,nn])]

        # 3) Calculate correlations
        r0=np.corrcoef(x,y)[0,1]
        corr=np.zeros(n)
        for nn in range(n):
            corr[nn]=metrics.r2_score(schufx[:,nn],schufy[:,nn])

        # 4) Params
        mu = np.mean(corr)
        std = np.std(corr)
        
        # 5) Quantiles
        lo = np.sort(corr)[round(len(corr)*pval)]
        hi = np.sort(corr)[round(len(corr)*(1-pval))]
       
        return mu, std, lo, hi

def varsMemory():
    import sys
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    out = sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
    return out

def reload(name):
    import importlib
    importlib.reload(name)

def mad(x):
    import numpy as np
    return np.mean(abs(x-np.median(x)))


def rss(fitted, obs):
    '''
    Residual Sum of Squares (RSS) between two functions.
    rss = rss(y, x)
    '''
    import numpy as np
    return np.sum((fitted-obs)**2)

def AIC(fitted, obs, k):
    '''
    Akaike's Information Criterion (AIC) based on Residual Sum of Squares (RSS)
    aic = AIC(fitted, obs, k)
    k is the number of intependent variables, INCLUDING the intercept in this case.
    '''
    import numpy as np
    n = len(obs)
    return n*np.log(rss(fitted, obs)/n) + 2*k

def mlr_sklearn(X,y,fit_intercept=True):
    '''
    fit, coefs, inter, r, aic, rmse = mlr_sklearn(X,y,fit_intercept=True)
    Simple and *fast* MLR with sklearn.    
    '''
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Take care of dimensions
    if np.size(np.shape(X))==1:
        nvars=1
        #X = X.reshape(-1,1)
        X = X[:,np.newaxis]
    else:
        nvars = X.shape[1]
    # MLR fit and results
    lr = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    coefs = lr.coef_
    inter = lr.intercept_
    fit = lr.predict(X)
    aic = AIC(fit, y, nvars)
    r = np.corrcoef(fit,y)[0,1]
    rms = rmse(fit,y) 
    return fit, coefs, inter, r, aic, rms

def bootsmlr_sklearn(X, y, n=10000, fit_intercept=True, stds=1, showProgress=True):
    '''
    coefs, inter, r, aic, rmse, fit, lo, hi = bootsmlr_sklearn(X, y, n=1000, fit_intercept=True)
    Returns the bootstrap distributins of MLR statistics.
    This runs relatively fast due to the simple implementation of sklearn.
    As used in: /home/scripts/python/acdtools/acdtools/11_MLRCEM_ProcessBasedSpatialModel_Compare.ipynb
    '''
    from tqdm.auto import tqdm
    import numpy as np
    import numpy.matlib
    import random
    
    # Check the number of covariates
    if np.size(np.shape(X))==1:
        nvars=1
        length=np.shape(X)[0]
    elif np.size(np.shape(X))>1:
        nvars=np.shape(X)[1]
        length=np.shape(X)[0]
        X=np.stack(X)
        
    # MLR on original order
    fit0, coefs0, inter0, r0, aic0, rmse0 = mlr_sklearn(X,y,fit_intercept=fit_intercept)
        
    # Create random indices    
    rand=np.zeros((length,n))
    for nn in range(n):
        for i in range(length):
            rand[i,nn]=random.randint(0,length-1) 

    # Sample data with replacement
    if nvars==1:
        schufx=np.zeros((length, n))
    else:
        schufx=np.zeros((length, n,nvars))
    schufy=np.zeros((length, n))
    for nn in range(n):
        for ii in range(length):
            if nvars==1:
                schufx[ii,nn]=X[int(rand[ii,nn])]
            else:
                schufx[ii,nn,:]=X[int(rand[ii,nn]),:]
            schufy[ii,nn]=y[int(rand[ii,nn])]
        
    # MLR on Sampled Data
    fit   = np.zeros((length,n))
    coefs = np.zeros((nvars,n))
    inter = np.zeros((n,))
    r     = np.zeros((n,))
    aic   = np.zeros((n,))
    rmse  = np.zeros((n,))
    if showProgress:
        for nn in tqdm(range(n)):
            if nvars==1:
                fit[:,nn], coefs[:,nn], inter[nn], r[nn], aic[nn], rmse[nn] = mlr_sklearn(schufx[:,nn], schufy[:,nn], fit_intercept=fit_intercept)
            else:
                fit[:,nn], coefs[:,nn], inter[nn], r[nn], aic[nn], rmse[nn] = mlr_sklearn(schufx[:,nn,:], schufy[:,nn], fit_intercept=fit_intercept)
    else:
        for nn in range(n):
            if nvars==1:
                fit[:,nn], coefs[:,nn], inter[nn], r[nn], aic[nn], rmse[nn] = mlr_sklearn(schufx[:,nn], schufy[:,nn], fit_intercept=fit_intercept)
            else:
                fit[:,nn], coefs[:,nn], inter[nn], r[nn], aic[nn], rmse[nn] = mlr_sklearn(schufx[:,nn,:], schufy[:,nn], fit_intercept=fit_intercept)
    
    # Standard deviations from Bootstrap Distributions
    coefs_std = stds*np.std(coefs, axis=1)
    inter_std = stds*np.std(inter)
    r_std     = stds*np.std(r)
    aic_std   = stds*np.std(aic)
    rmse_std  = stds*np.std(rmse)
    
    # Set up all possible combination of signs
    signs = np.zeros((2**(nvars+1),nvars+1))
    for j in range(nvars+1):
        count=((2**(nvars+1))/2**(j+1))
        fir=np.zeros((int(count),)); fir.fill(1)
        sec=np.zeros((int(count),)); sec.fill(-1)
        signs[:,j]=np.matlib.repmat(np.hstack([fir,sec]),1,2**(j)) 

    # Make all possible lines, combining the signs of uncertainties
    XX=X.copy()
    lines=np.zeros((len(y),2**(nvars+1)))
    for i in range(2**(nvars+1)):
        if nvars==1:
            lines[:,i] = inter0 + signs[i,0]*inter_std + np.squeeze(XX*(coefs0[0] + signs[i,1]*coefs_std[0]))
        else:
            linterm = inter0 + signs[i,0]*inter_std
            angterm = 0
            for nn in np.arange(1,nvars+1):
                angterm = angterm + XX[:,nn-1]*(coefs0[nn-1]+signs[i,nn]*coefs_std[nn-1])
            lines[:,i] = linterm + angterm
    combs=np.stack(lines,axis=0)
    lo=np.min(combs, axis=1)
    hi=np.max(combs, axis=1)
    
    
    return coefs, inter, r, aic, rmse, fit, lo, hi

def bootstats(x, y=0, stats='r', n=1000, length=0, k=2):
    '''
    out = bootstats(x, y, stats='r')
    
    Bootstrapping of sample statistics. `stats` options are: r, aic, rmse, mean, std.
    For aic, the number of parameters (INCLUDING the intercept) k is needed (default is k=2).
    For mean and std, means and standard deviations are sampled from x. y is ignored.
    '''
    import random
    import numpy as np
    if length == 0:
        length = len(x)

    # Generate random positions
    
    rand=np.zeros((length,n))
    for nn in range(n):
        for i in range(length):
            rand[i,nn]=random.randint(0,len(x)-1) # must sample from entire length

    # Shuffle time series (actually, sample with replacement)             
    schufx = np.zeros((length,n))
    if stats!='mean' and stats!='std':
        schufy = np.zeros((length,n))
    for nn in range(n):
        for ii in range(length):
            schufx[ii,nn] = x[int(rand[ii,nn])]
            if stats!='mean' and stats!='std':
                schufy[ii,nn] = y[int(rand[ii,nn])]
    
    # Calculate statistics
    out = np.zeros(n)
    for nn in range(n):
        if stats=='r':
            out[nn] = np.corrcoef(schufx[:,nn],schufy[:,nn])[0,1]
        elif stats=='aic':
            out[nn] = AIC(schufx[:,nn],schufy[:,nn],k)
        elif stats=='rmse':
            out[nn] = rmse(schufx[:,nn],schufy[:,nn])
        elif stats=='mean':
            out[nn] = np.mean(schufx[:,nn])
        elif stats=='std':
            out[nn] = np.std(schufx[:,nn])
        else:
            print('Invalid stats. Options are:')
            print('r, aic, rmse, mean, std')
            return
    return out

def loo_predict_sklearn(X, coefs, inter):
    import numpy as np
    nvars = coefs.shape[0]
    out = inter
    for i in range(nvars):
        out = out + X[i]*coefs[i]
    return out

