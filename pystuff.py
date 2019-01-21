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


Found a bug? Please let me know:
davidnielsen@id.uff.br
"""
def bootscorr(x, y, n=10000, conflev=0.95, positions='new',details=False):

    """
    IN:
        x: numpy array, time series
        y: numpy array, time series
        n: number of bootstrap iterations (default = 10k)
        conflev: you get it.
        positions: if "new", random pairwise positions (with replacement) will be generated.
        Otherwise, supply with numpy array with positions (same length as x and y, n-dimension in the columns). 
        details: Boolean, controls the output (defauls=False)
    
    OUT:
        if details==False: 
            r0  = linear pearson correlation coefficient between x and y [-1,1]
            lev = minimum significance level for which r0 is different than zero [0,1]
    
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
            sig=1
        else:
            sig=0
        
        # 5) Check all possible p-values within n to get minimum significance level (minsig)
        minsig=np.nan
        tails=np.arange(0.01,0.5,0.00001)
        for i in range(len(tails)):
            if np.isnan(minsig):
                tail=tails[i]/2
                qinf=round(n*tail)
                qsup=round(n-qinf)
                rrinf=sort[int(qinf)]
                rrsup=sort[int(qsup)]
                if rrinf>0 or rrsup<0:
                    minsig=tail*2
                    lev=(1-minsig)
        
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
    
    # Get PC's
    scores = mydata2 @ eigenvecs
    
    # Explained variance
    expl=eigenvals*100/sum(eigenvals)
    expl_acc=expl.copy()
    for i in np.arange(1,len(expl)):
        expl_acc[i]=expl[i]+expl_acc[i-1]
    
    return scores, eigenvals, eigenvecs, expl, expl_acc, means, stds

##### 3) Remove Linear Trend #####

def ddetrend(var,xvar=321943587416321,returnTrend=False):

    """
    14.01.2019
    """

    import numpy as np
    from scipy import stats

    # If x is not given, use sequence
    if type(xvar)==int and xvar==321943587416321:
        xvar=np.arange(0,len(var))
        print(np.shape(xvar))
        print(np.shape(var))

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

    # Make linear model, calculate and remove trend
    slope, intercept, _, _, _ = stats.linregress(xvar_clean,var_clean)
    trend=np.array(xvar_clean)*slope+intercept
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

def getbox(coords,lat,lon,data):
    
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

    return inbox, meanbox

########## 5) Running mean

def runmean(x,window=3,fillaround=False):
    
    '''
    Calculates a running mean of a given series x, using a window
    of given length. The window must be odd, ideally. Otherwise, 
    an approximation will be made.
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
    
    # This option will apply increasing windows on borders
    # so that the len(outseries)=len(inseries)
    if fillaround:
        increasingWindows=np.arange(3,window+1,2)
        print(increasingWindows)
        x_rm=np.zeros(np.shape(x))
        for w in range(len(increasingWindows)):
            halfwindow=int((increasingWindows[w]-1)/2)
            print(halfwindow)
            for t in range(len(x)):
                if t>=halfwindow and t<(len(x)-halfwindow):
                    x_rm[t]=np.mean(x[t-halfwindow:t+halfwindow],axis=0)
                else:
                    if halfwindow==1:
                        x_rm[t]=x[t]
                    else:
                        x_rm[t]=x_rm[t]
    else:
        x_rm=np.zeros(np.shape(x))
        halfwindow=int((window-1)/2)
        for t in range(len(x)):
            if t>=halfwindow and t<(len(x)-halfwindow):
                x_rm[t]=np.mean(x[t-halfwindow:t+halfwindow],axis=0)
            else:
                x_rm[t]=np.nan
                
    return x_rm

#################### 6) ddreg
    
def ddreg(x,y):
    from scipy import stats
    slope, inter, _, _, _ = stats.linregress(x,y)
    trend=np.array(x)*slope+inter
    return trend


