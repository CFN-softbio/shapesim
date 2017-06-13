#quick optics calculations
import numpy as np

'''Library to perform some quick calculations for the xray optics
    Example:
        get speckle size (in inverse Angs) for a 1 micron beam:
            calc_specklesize(1)
        get q spanned (in inverse Angs) by a 75e-6 m pixel at 15 meters for a 1 Angstrom beam:
            calc_q(15.,75e-6,1.)
        calculate the number of pixels per speckle for this setup:
            calc_specklesize(1)/calc_q(15.,75e-6,1.)
        A 20 times difference
'''

#Thomson radius
THOMSON_R = 2.8179403267e-5

def calc_tth(q, wv):
    ''' calculate two theta from q.'''
    tth = 2*np.arcsin(wv*q/4/np.pi)
    return tth

def calc_qy(wv,ttheta):
    ''' calculate the out of plane component of the detector
        for a certain angle and wavelength.'''
    return 2*np.pi/wv*(1 - np.cos(ttheta))

def calc_qz(wv,ttheta):
    ''' calculate the in plane component of the detector for a certain angle
            and wavelength.'''
    return 2*np.pi/wv*(np.sin(ttheta))


def calc_specklesize(w):
    '''Calculate the approximate speckle size in q from a random sample of
        dimensions w x w.
        w - dimensions in microns
        returns speckle size in inverse angstroms
'''
    return 2*np.pi/(w*1e4)

def calc_q(L,a,wv):
    ''' calc_q(L,a,wv) - calculate the q value for length L, transverse
            distance a and wavelength wv.
        Use this to calculate the speckle size

        L - sample to detector distance (mm)
        a - pixel size transverse length from beam direction (mm)
        wv - wavelength
        Units of L and a should match and resultant q is in inverse units of wv.
       '''
    theta = np.arctan2(a,L)
    q = 4*np.pi*np.sin(theta/2.)/wv
    return q

def calc_detdistance(L,a,wv):
    ''' calc_detdistance(qpp,a,wv) - calculate the q value for length L, transverse
            distance a and wavelength wv.
        Use this to calculate the speckle size

        L - sample to detector distance (mm)
        a - pixel size transverse length from beam direction (mm)
        wv - wavelength
        Units of L and a should match and resultant q is in inverse units of wv.
       '''
    theta = np.arctan2(a,L)
    q = 4*np.pi*np.sin(theta/2.)/wv
    return q

def calc_ddq(L,a,wv):
    ''' Calculate the delta delta q. By this, I mean the deviation from the simulation
        using the Fourier transform the to the actual q value on the detector, which
        probes an Ewald sphere, not a linear q space.'''
    return 4*np.pi/wv*(a/2./L-np.sin(np.arctan2(a,L)/2.))

def calc_ddqq(L,a,wv):
    ''' Calculate the delta delta q/q. By this, I mean the deviation from the simulation
        using the Fourier transform the to the actual q value on the detector, which
        probes an Ewald sphere, not a linear q space.
        This one is normalized by the q value on the Ewald sphere (so normalized error).
    '''
    return 4*np.pi/wv*(a/2./L-np.sin(np.arctan2(a,L)/2.))/calc_q(L,a,wv)

def calc_detflux(V,drhoe, flux, L,a):
    ''' calc_detflux(V,drhoe, flux, L,a)
            Calculate the flux at the detector:
            V - volume of sample
            drhoe - electron density difference
            L - sample to detector distance
            a - transverse distance from beam
        This will be flux at q=0 (not including straight through beam)
        You then need to multiply this by the |F(q)|^2 normalized to 1 at q=0
        for the actual flux.
    '''
    dOmega = np.arctan2(a,L)**2
    res = (THOMSON_R*V*drhoe)**2/flux/dOmega
    return res

def calc_longcoh(wv,wvres):
    '''
        wv : wavelength
        wvres : wavelength resolution (delta Lambda/lambda)
    '''
    return .5*wv/wvres

def calc_trancoh(L,a,qv):
    '''calc_trancoh(L,a) Calculate the transverse coherence length

        L : source to sample length
        a : source size
    '''
    return .5*wv*L/a

def calc_pathlengthdiff(L,a,s):
    '''
        Calculate the path length difference of waves traveled in the longitudinal direction
            for a typical SAXS setup
        L - sample to detector distance
        a - transverse length
        s - sample thickness
    '''
    #first get path length difference
    return .5*s*a**2/L**2

def calc_maxsamplethickness(wv,wvres,L,a):
    ''' Calculate the maximum sample thickness (in mm) that can be used before
            coherence is lost.
        wv - wavelength (in Angstroms)
        wvres - wavelength resolution (delta lambda/lambda)
        L - sample to detector distance (same units as det height)
        a - maximum detector height (same units as det distance)
        returns maximum sample thickness in millimeters
    '''
    Lcoh = calc_longcoh(wv,wvres)
    return 2*(Lcoh*1e-7)*(L/a)**2
