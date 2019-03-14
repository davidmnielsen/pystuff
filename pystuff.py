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
    
Found a bug? Please let me know:
davidnielsen@id.uff.br
"""
def bootscorr(x, y, n=10000, conflev=0.95, positions='new',details=False):

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

def ddetrend(var,xvar=321943587416321,returnTrend=False,center=False):

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

def runmean(x,window=3,fillaround=False,weights=False):
    
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
                            x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow]*\
                                           weights[-2*halfwindow-1:-1],
                                           axis=0)/np.sum(weights[-2*halfwindow-1:-1])
                        elif t>=(len(x)-len(weights)):
                            # t is on the right edge
                            x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow]*\
                                           weights[:2*halfwindow],
                                           axis=0)/np.sum(weights[:2*halfwindow])
                        else:
                            # t is not on the edges
                            midw=(len(weights)/2)+1
                            x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow]*\
                                           weights[midw-halfwindow:midw+halfwindow],axis=0)/\
                                           np.sum(weights[midw-halfwindow:midw+halfwindow])                       
                    else:
                        # t not on the edges without weight
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
                    x_rm[t]=np.mean(x[t-halfwindow:t+halfwindow+1],axis=0)
                else:
                    x_rm[t]=np.sum(x[t-halfwindow:t+halfwindow+1]*weights,axis=0)/np.sum(weights)
            else:
                x_rm[t]=np.nan                
    return x_rm


#################### 6) ddreg
    
def ddreg(x,y,returnStats=False):
    import numpy as np
    from scipy import stats
    slope, inter, r, p, se = stats.linregress(x,y)
    trend=np.array(x)*slope+inter
    if returnStats:
        return slope, r, p    
    else:    
        return trend

################## 7) Standardize

def standardize(x,center=True):
    import numpy as np
    xn = np.full(np.shape(x),np.nan)
    if center:
        xn = (x - np.nanmean(x))/np.nanstd(x)
    else:
        xn = x/np.nanstd(x)
    return xn

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

def rednoise(lenx, rho, nsim=1000, dist='normal', returnWhite=False):
    # Creates nsim time series of rednoise of length=lenx, with lag-1 autocorrelation rho.
    import numpy as np
    srho=(1-(rho**2))**(0.5)
    red=np.zeros((lenx,nsim))
    white=np.zeros((lenx,nsim))
    for j in range(nsim-1):
        for i in range(lenx-1):
            if dist=='normal':
                #white[i+1,j]=white[i,j]+np.random.normal() # mu=0, std=1
                white[i+1,j]=np.random.normal() # mu=0, std=1
            elif dist=='uniform':
                #white[i+1,j]=white[i,j]+np.random.uniform(-1,1)
                white[i+1,j]=np.random.uniform(-1,1)
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

def theored(dt,rho,meanP,f):
    import numpy as np
    fnyq=1/(2*dt)
    theo=np.zeros((len(meanP,)))
    for i in range(len(meanP)):
        theo[i]=(1-rho**2)/(1-(2*rho*np.cos(np.pi*f[i]/fnyq))+rho**2)
    theoun=theo[0]
    theo[0]=0
    Art=np.sum(theo)/(len(meanP))
    theo[0]=theoun
    Ax=np.sum(meanP)/len(meanP)
    theo=theo*(Ax/Art);
    return theo

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
    weights    = low_pass_weights(windowlen, 1/cutoff)
    xlow       = runmean(x,window=len(weights),weights=weights,fillaround=fillaround)
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
    
    # Calculate nsim red-noise time series
    red =rednoise(len(x), rhoAlt(x,dt), nsim ,dist='uniform')
    
    # Calculate periodograms of the nsim red-noise series
    fn=np.zeros((len(f),nsim))
    Pn=np.zeros((len(f),nsim))
    for n in range(nsim):
        fn[:,n], Pn[:,n] = signal.periodogram(red[:,n],fs=dt,detrend='linear')
        
    # Mean spectrum of nsim simulations
    meanP=np.mean(Pn,axis=1)
    
    # Calculate theoretical mean spectrum of nsim simulations
    theo = theored(1/dt,rhoAlt(x,dt),meanP,f)
    
    # Calculates confidence curves
    nw=5
    coefs=np.zeros((100,))
    conflevs=np.zeros((100,len(f)))
    for c in range(100):
        coefs[c]=chi2.ppf(c/100, 2*(2*nw-1))/(2*(2*nw-1))
        conflevs[c,:]=coefs[c]*theo
        
    return f, psd, conflevs, max5, meanP 
    
    
