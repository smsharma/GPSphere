import numpy as np

def calc_dist_chord(ra_ref, dec_ref, ra, dec):

    return 2*np.sqrt(np.sin((dec - dec_ref)/2)**2
                     + np.cos(dec_ref)*np.cos(dec)*np.sin((ra - ra_ref)/2)**2)

def calc_dist(ra_ref, dec_ref, ra, dec):

    """

    Function calcdist

    Distance between a point raref, decref and an array ra, dec.  All
    positions should be in radians, and ra and dec should have
    matching shapes.
    
    Inputs:
    ra_ref: float, reference right ascension in radians
    dec_ref: float, reference declination in radians
    ra: array, list of right ascensions in radians
    dec: array, list of declinations in radians

    Returns:
    an array of distances, same shape as ra, measured in radians

    """
    
    return np.arccos(np.sin(dec)*np.sin(dec_ref) +
                     np.cos(dec)*np.cos(dec_ref)*np.cos(ra_ref - ra))


def calc_all_dist(ra1, dec1, ra2, dec2, chord=False):

    """.
    
    Function calc_all_dist.

    Computes distances between all pairs of points in the 1d arrays
    ra1, dec1.  Outputs a 2D array of size n x m, where n is the
    dimensionality of ra1, and m is the dimensionality of ra2.

    Inputs:
    ra1: array of ra (in radians)
    dec1: array of dec (in radians)
    ra2: second array of ra
    dec2: second array of dec

    Returns: 
    dist: 2d array of distances between the points in ra1, dec1 and ra2, dec2, in radians

    """
    
    dist = np.zeros((ra1.shape[0], ra2.shape[0]), np.float32)
    for j in range(ra1.shape[0]):
        if chord:
            dist[j] = calc_dist_chord(ra1[j], dec1[j], ra2, dec2)
        else:                
            dist[j] = calc_dist(ra1[j], dec1[j], ra2, dec2)
    return dist


def calcbestdist(ralist, declist, ra, dec):

    """

    Function calcbestdist

    Find the closest point in the input lists ralist and declist to
    each position defined by the input arrays ra and dec.  

    Inputs:
    ralist: array-like, list of reference right ascensions in radians
    declist: array-like, list of reference declinations in radians
    ra: array-like, list of right ascensions in radians
    dec: array-like, list of declinations in radians

    Outputs:
    closest_pt: array giving the index of the closest element in ralist, declist for each pair of elements ra, dec
    bestdist: array giving the distance (in radians) of the closest element in ralist, declist for each pair of elements ra, dec
    n: array giving the number of stars that are closer to each element in ralist, declist than to any other element

    """

    bestdist = np.zeros(ra.shape) + 1e100
    closest_pt = np.zeros(ra.shape).astype(int)
    
    for i in range(len(ralist)):
        dist = calc_dist(ralist[i], declist[i], ra, dec)
        indx = np.where(dist < bestdist)
        closest_pt[indx] = i
        bestdist[indx] = dist[indx]

    n = np.zeros(ralist.shape).astype(int)
    
    for i in range(len(ralist)):
        n[i] = np.sum(closest_pt == i)

    return [closest_pt, bestdist, n]
