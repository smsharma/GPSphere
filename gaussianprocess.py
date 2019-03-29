# Gaussian processes on a sphere
# Code originally from Tim Brandt (UCSB)

import numpy as np
from astropy.io import fits
import distances
from scipy import special, interpolate, optimize
import time

def like_gaussmix(x, val1, var1):
    
    pmra = val1
    dpmra = var1
        
    lnL = np.sum(-(val1 - x)**2/(2*dpmra**2))

    return lnL

def explorelike(p, d, d_cross, logd, logd_cross,
                val1, var1):

    sig2, alpha, nu = p
        
    if sig2 <= 0 or alpha <= 0 or nu <= 0 or nu > 0.5:
        return 1e100
    
    x = gaussianprocess(d, d_cross, logd, logd_cross, val1, var1,
                        sig2, alpha, nu)

    return -like_gaussmix(x, val1, var1)

def geninterp(xmin, xmax, sig2, nu, tol=1e-5):

    n = 21
    x = np.linspace(np.log(xmin), np.log(xmax), n)
    y = sig2*2**(nu - 1)/special.gamma(nu)*np.exp(x*nu)
    y *= special.kv(nu, np.exp(x))
    f = interpolate.interp1d(x, y, bounds_error=False)
    _x = x.copy()
    _y = y.copy()
    
    for i in range(15):
        n = (n//2)*4 + 1
        x = np.linspace(np.log(xmin), np.log(xmax), n)[1:-1:2]
        y = sig2*2**(nu - 1)/special.gamma(nu)*np.exp(x*nu)
        y *= special.kv(nu, np.exp(x))
        indx = np.where((f(x) - y)**2 > tol**2)
        if len(indx[0]) == 0:
            break
        _x = np.asarray(list(_x) + list(x[indx]))
        _y = np.asarray(list(_y) + list(y[indx]))
        f = interpolate.interp1d(_x, _y, bounds_error=False)
    return f

def covariance(d, sig2, alpha, nu, logd, tol=1e-6):

    mindist = 0.0000166
    f = geninterp(mindist/alpha, np.pi/alpha, sig2, nu, tol=tol)
    # dx = 1e-3
    # x = np.arange(np.log(mindist/alpha), np.log(np.pi/alpha) + dx, dx)
    # y = sig2*2**(nu - 1)/special.gamma(nu)*np.exp(x*nu)
    # y *= special.kv(nu, np.exp(x))
    # f = interpolate.interp1d(x, y, bounds_error=False)
    covar = f(logd - np.log(alpha))
    eps = 1e-10
    val = sig2*2**(nu - 1)/special.gamma(nu)*(eps)**nu*special.kv(nu, eps)
    covar[np.where(d == 0)] = val        

    return covar
    
def gaussianprocess(d, d_cross, logd, logd_cross, y, var,
                    sig2, alpha, nu, tol=1e-6):
    
    K11 = covariance(d, sig2, alpha, nu, logd, tol=tol) + np.identity(d.shape[0])*var
    K11_inv = np.linalg.inv(K11)
    K12 = covariance(d_cross, sig2, alpha, nu, logd_cross, tol=tol)
    return np.linalg.multi_dot([K12, K11_inv, y])
