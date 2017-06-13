# scattering simulator
import numpy as np
from scipy.stats import expon, lognorm, gamma, norm, uniform
from misc import constants

from ..shapes import Shape3, Annuli3, Tile3, Nmer3, Shape3Spheres, \
        Shape3, HexLattice3Spheres, Shape3Superballs
from ..tools.matrices import rotmat3D
from ..tools.optics import calc_q
from ..tools.smooth import smooth2Dgauss

class Simulator:
    def __init__(self, Nparticles, *args, avoidance=None, rho=1.,dims=[1000,1000,1000],resolution=None,
                 unit='rad*pixel',margin=10,maxr=None,PF=0, mu=np.nan,mubg=None,smoothingfac=None, incoherent=False,\
                 sticky=[None, None, None], mask=None, **kwargs):
        ''' Initialize the simulator.
            Initialize the simulator
        Simulate something, or any number of them.  Does translations or
        rotations in plane only.  avoidance is the avoidance distance between
        spheres
        margin - margin that particles must be in box ( in pixels) note :
        margin has been changed to a circular area The radius of the circular
        area is equal to Nd/2-margin
        mu : if specified Poissonize the image with this mu factor
        maxr : maximum radius (in pixels, not units specified) of the box
        sticky : the stickyness potential.
                It is defined as the following list:
                    [stickyc, stickyr, stickyrng]
                    stickyc - the chance of stickiness
                    stickyr - the effective radius of stickiness
                    stickytype - the type of stickiness
                Think of it as a uniform prob distribution plus sticky:
                    x P_u(x) + (1-x)*P_r(x,r) <-- sticky part
                see function _stickyrngfromtype for types available (currently 'expon', 'lognorm', 'normal')
            NOTE: units should match the units used in the simulation
            NOTE : for now, resolution doesn't work (just use everything as pixels). should be corrected
                later
        xrdata : an xray data object. Must contain the following:
            xrdata.dims
            xrdata.qperpixel (assumes Ewald curvature negligible)
            xrdata.unit (default to inv angs)
            xrdata.x0
            xrdata.y0

            arguments : Nparticles + other params
        '''

        # this will call the Nmer3 or other shape because of the MRO
        super(Simulator,self).__init__(*args,dims=dims, resolution=resolution, unit=unit,\
                                rho=rho, avoidance=avoidance,**kwargs)
        self.incoherent = incoherent
        self.Nparticles = Nparticles
        self.smoothingfac = smoothingfac
        # cludgy fix just to test radius instead of margin (to remove box symmetry in correlations
        # for large margins)
        self.computebounds(margin,maxr=maxr)
        self.set_sticky(sticky)
        self.set_mu(mu)
        self.set_mubg(mubg)
        self.set_mask(mask)

        self.rotbounds = 2*np.pi
        self.maxtries = np.nan
        self.fimg2_incoh = None
        self.PF = PF

    def getNQBOX(self, DIMS,XCEN,YCEN):
        ''' Get the square dimensions and the subselection of the box
            for the simulation necessary.'''
        N = np.max(DIMS)*2

        DX = N//4 - XCEN + 1
        DY = N//4 - YCEN + 1
        # [x0, x1, y0, y1]
        IMGQBOX = np.array([DX+ N//4, DIMS[1]+DX+ N//4, DY+ N//4, DIMS[0]+DY+ N//4])

        return N, IMGQBOX

    def getresolution(self, qperpixel, N):
        ''' Get the resolution necessary in Angstroms per pixel an image of
            dimension N to have a resolution in the Fourier domain of N
            of qperpixel.'''
        return 2*np.pi/N/qperpixel

    def getscat(self):
        if self.IMGQBOX is not None:
            x0,x1,y0,y1 = self.IMGQBOX.astype(int)
            return self.fimg2[y0:y1,x0:x1]
        else:
            return self.fimg2

    def set_maxr(self,maxr):
        self.computebounds(maxr=maxr)

    def set_mu(self, mu):
        ''' If mu is None, then remove poisson'''
        self.mu = mu
        if mu is None or np.isnan(mu):
            self.poisson = False
        else:
            self.poisson = True

    def set_ld(self,ld):
        self.ld = ld
        self.reset()

    def set_mubg(self, mubg):
        ''' set background. NOTE: this does not
            necessarily set the poisson term to true.'''
        self.mubg = mubg

    def set_mask(self,mask):
        self.mask = mask

    def set_avoidance(self, avoidance):
        self.avoidance = avoidance

    def set_Nparticles(self, Nparticles):
        self.Nparticles = Nparticles

    def set_smoothingfac(self, smoothingfac):
        self.smoothingfac = smoothingfac

    def set_coherence(self, coherence):
        ''' True for coherent False for incoherent.'''
        self.incoherent = not coherence

    def set_sticky(self, sticky):
        ''' Set the sticky chance stickyc, radius, stickyr,
            and type, stickytype.
            See _stickyrngfromtype for list of types possible
            if stickytype doesn't exist, stickyrng gets set to
                None and no potential is used (stickyr and stickyc kept
                but ignored)
            NOTE: some precision will be lost when going from string to int
                but this is a shortcut that makes the programming easier
        '''
        if sticky[0] is not None and sticky[0] != 'None':
            self.stickyc = float(sticky[0])
        else:
            self.stickyc = None
        if sticky[1] is not None and sticky[1] != 'None':
            self.stickyr = float(sticky[1])
        else:
            self.stickyr = None
        self._stickyrngfromtype(sticky[2],self.stickyr)

    def _stickyrngfromtype(self,typestr,stickyr):
        ''' Set the sticky type. Currently, three probability distributions
            are accepted:
            expon : exponential
            lognorm : lognormal
            normal : normal
        '''
        if typestr == 'expon':
            self.stickyrng = expon(0,stickyr)
        elif typestr == 'lognorm':
            # lognorm : 1, 0, 1 first is shape (keep as 1)
            # second is the location/shift (keep at 0)
            # third is going to peak at half the radius (by eye I checked approximately
            # that's what's happening. will need to be more thorough later)
            self.stickyrng = lognorm(1,0,stickyr*2)
        elif typestr == 'norm':
            # normal distribution
            self.stickyrng = norm(0, stickyr)
        elif typestr == 'uniform':
            # uniform distribution
            self.stickyrng = uniform(0, stickyr)
        else:
            self.stickyrng = None

    def computebounds(self,margin=None, maxr=None):
        ''' compute the bounds from the margin.
                specify maxr as a flag
        '''
        if maxr is None:
            self.margin = margin
            # making this backwards compatible with previous code
            if hasattr(self,'ld'):
                ld = self.ld
            else:
                ld = 0
            self.tranboundsx = (self.dims[0]/2.*self.resolution-ld-self.margin*self.resolution)
            self.tranboundsy = (self.dims[1]/2.*self.resolution-ld-self.margin*self.resolution)
            self.maxr = self.tranboundsx
        else:
            self.tranboundsx = maxr*self.resolution
            self.tranboundsy = maxr*self.resolution
            self.maxr = maxr*self.resolution

    def checkinbounds(self,vec):
        ''' check if a vector is within bounds.'''
        return np.sqrt(vec[0]**2 + vec[1]**2) <= self.maxr

    def setflux(self, flux=None):
        ''' Set the flux of the simulator'''
        self.set_mu(flux)

    def addOne(self, position, phi=None):
        ''' Need to define this.'''
        pass

    def randomize(self, orient=False,project=True):
        ''' Make a new random instance.
            if project is false, don't project image (only for coherent case),
                saves time to generate vectors
        '''
        self.clearunits()
        rndr = np.random.random(1)
        phi = rndr*self.rotbounds
        if self.incoherent is True:
            if(self.fimg2_incoh is not None):
                self.fimg2_incoh *= 0
            else:
                self.fimg2_incoh = np.zeros((self.dims[0], self.dims[1]))
        for i in range(self.Nparticles):
            if self.incoherent:
                self.clearunits()
            res = -1
            cnter = 0

            stick = False
            if self.stickyrng is not None and i != 0:
                if self.stickyc >= np.random.random():
                    stick = True
                    stickno = np.floor(np.random.random()*i).astype(int)
                    stickcnter = 0
                else:
                    stick = False

            while(res == -1):
                while True:
                    if stick:
                        stickcnter += 1
                        if stickcnter > 10:
                            # most likely there is no space here, try somewhere else
                            stickcnter = 0
                            stickno = np.floor(np.random.random()*i).astype(int)
                    rndr = np.random.random(3)
                    if orient is False:
                        phi = rndr[0]*self.rotbounds
                    if stick is False:
                        vecx = (rndr[1]-.5)*2*self.tranboundsx
                        vecy = (rndr[2]-.5)*2*self.tranboundsy
                    else:
                        # limit particle movement to surround ith particle
                        # uniform theta
                        vecr = np.abs(self.stickyrng.rvs())
                        vecth = np.random.random()*2*np.pi
                        vecx = vecr*np.cos(vecth) + self.Nmervecs[stickno][0]
                        vecy = vecr*np.sin(vecth) + self.Nmervecs[stickno][1]
                    # always check that it's within bounds
                    if np.sqrt(vecx**2 + vecy**2) > self.maxr:
                        cnter += 1
                        continue
                    else:
                        cnter += 1
                        break
                tranvec = np.array([vecx, vecy, 0])
                res = self.addOne(tranvec,phi=phi)
                if(cnter > 1):
                    if self.PF == 1:
                        print("failed once")
                if(cnter > self.maxtries):
                    if self.PF == 1:
                        print("Error, max tries exceeded for overlap avoidance: {}".format(self.maxtries))
                        print("Returning actual number of particles added")

                    return i+1
            if self.incoherent is True:
                self.project()
                self.fimg2_incoh += self.fimg2

        if self.incoherent is True:
            self.fimg2 = np.copy(self.fimg2_incoh)

        if project:
            self.project_sim()

    def project_sim(self):
        if self.incoherent is False:
            self.project();

        if self.mubg is not None:
            if not np.isnan(self.mubg).any():
                self.fimg2 += self.mubg

        if(self.poisson):
            # simulate Poisson scattering
            self.fimg2 = np.random.poisson(self.mu*self.fimg2).astype(float)

        if self.smoothingfac is not None:
            self.fimg2 = smooth2Dgauss(self.fimg2,mask=self.mask,sigma=self.smoothingfac)

        if self.mask is not None:
            self.fimg2 *= self.mask



# Method resolution order (Nmer3Simulator -> Simulator -> Nmer3 -> Shape3Spheres -> Shape)
class Nmer3Simulator(Simulator, Nmer3):
    '''

        args order : radius, ld, n
    '''
    def __init__(self, Nparticles, radius, ld, n, avoidance=None, rho=1.,dims=[1000,1000,1000],resolution=None,
                 unit='rad*pixel',margin=10,maxr=None,PF=0, mu=np.nan,smoothingfac=None, incoherent=False,\
                 sticky=[None, None, None],xrdata=None,mask=None,mubg=None):
        ''' Initialize the simulator.'''
        # based on MRO, this will call Nmer3
        super(Nmer3Simulator,self).__init__(Nparticles, radius,ld,n,rho=rho,dims=dims, resolution=resolution, \
                            unit=unit,avoidance=avoidance, mask=mask, xrdata=xrdata,mubg=mubg, margin=margin,\
                            maxr=maxr,sticky=sticky,mu=mu,smoothingfac=smoothingfac)
        self.addOne = self.addNmer

# Method resolution order (Nmer3Simulator -> Simulator -> Nmer3 -> Shape3Spheres -> Shape)
class Tile3Simulator(Simulator, Tile3):
    '''

        args order : radius, ld, tilebin
        if len(Nparticles) > 1 and its length equals the number of tilebins, then
            assume this is the numbers of tiles per tile bin
        else assume it's the number of each tile
    '''
    def __init__(self, Nparticles, radius, ld, tilebins, avoidance=None, rho=1.,dims=[1000,1000,1000],resolution=None,
                 unit='rad*pixel',margin=10,maxr=None,PF=0, mu=np.nan,smoothingfac=None, incoherent=False,\
                 sticky=[None, None, None],xrdata=None,mask=None,mubg=None):
        ''' Initialize the simulator.'''
        # based on MRO, this will call Nmer3
        if tilebins.ndim==2:
            tilebins = tilebins.reshape(1, tilebins.shape[0],tilebins.shape[1])
        Nparticles = np.array(Nparticles).reshape(-1)
        if len(Nparticles) == 1:
            Nparticles = np.tile(Nparticles,tilebins.shape[0])
        Nparticles_tot = np.sum(Nparticles)
        self.Nparticles_csum = np.cumsum(Nparticles)
        self.Nparticles_csum = np.concatenate(([0],self.Nparticles_csum))
        super(Tile3Simulator,self).__init__(Nparticles_tot, radius,ld,tilebins,rho=rho,dims=dims, resolution=resolution, \
                            unit=unit,avoidance=avoidance, mask=mask, mubg=mubg,xrdata=xrdata)
        self.curtilenumber = 0


    def addOne(self,vec0,phi=0.):
        if self.curtilenumber < self.Nparticles_csum[-1]:
            tiletype = np.where(self.curtilenumber < self.Nparticles_csum)[0][0]-1
            res = self.addTile(vec0,phi=phi, tiletype=tiletype)
            if res != -1:
                self.curtilenumber += 1
            return res
            print("tile type: {}".format(tiletype))
        else:
            # do nothing, return some val not -1
            return 0

    def clearunits(self):
        super(Tile3Simulator,self).clearunits()
        self.curtilenumber = 0


# Method resolution order (Nmer3Simulator -> Simulator -> Shape3Superballs-> Shape)
class Superball3Simulator(Simulator, Shape3Superballs):
    '''

        args order : radius, ld, n
    '''
    def __init__(self, Nparticles, radius, ld, n=1,avoidance=None, rho=1.,dims=[1000,1000,1000],resolution=None,
                 unit='rad*pixel',margin=10,maxr=None,PF=0, mu=np.nan,smoothingfac=None, incoherent=False,\
                 sticky=[None, None, None],xrdata=None,mask=None,mubg=None, **kwargs):
        ''' Initialize the simulator.'''
        # based on MRO, this will call Nmer3
        super(Superball3Simulator, self).__init__(Nparticles, radius,ld,n,rho=rho,dims=dims, resolution=resolution, \
                            unit=unit,avoidance=avoidance, mask=mask, xrdata=xrdata,mubg=mubg,**kwargs)
        self.addOne = self.addOnetmp
        self.radius = radius
        self.p = ld
        self.n = n

    def addOnetmp(self,vec0,phi=0.,rho=1.):
        self.addunits([vec0])

# Method resolution order (Hex3Simulator -> Simulator -> HexLattice3Spheres -> Shape3Spheres -> Shape)
class Hex3Simulator(Simulator, HexLattice3Spheres):
    ''' Simulate a Hex, or any number of them.
        Does translations or rotations in plane only.
        avoidance is the avoidance distance between spheres
        margin - margin that particles must be in box ( in pixels)
            note : margin has been changed to a circular area
            The radius of the circular area is equal to Nd/2-margin
        mu : if specified Poissonize the image with this mu factor
        Narray : num elements in array
    '''
    def __init__(self, Nparticles, radius, ld, Narray, avoidance=None, rho=1.,dims=[1000,1000,1000],resolution=None,
                 unit='rad*pixel',margin=10,maxr=None,PF=0, mu=np.nan,smoothingfac=None, incoherent=False,\
                 sticky=[None, None, None],xrdata=None, mask=None,mubg=None):
        ''' Initialize the simulator.'''
        # based on MRO, this will call Nmer3
        super(Hex3Simulator,self).__init__(Nparticles, radius,ld, Narray,rho=rho,dims=dims, \
                        resolution=resolution, unit=unit,avoidance=avoidance, mask=mask, xrdata=xrdata, mubg=mubg,mu=mu)
        self.addOne = self.addHex

# Method resolution order (Annuli3Simulator -> Simulator -> Annuli3 -> Shape)
class Annuli3Simulator(Simulator, Annuli3):
    ''' Simulate Annuli, or any number of them.
        Does translations or rotations in plane only.
        avoidance is the avoidance distance between spheres
        margin - margin that particles must be in box ( in pixels)
            note : margin has been changed to a circular area
            The radius of the circular area is equal to Nd/2-margin
        mu : if specified Poissonize the image with this mu factor
        Narray : num elements in array
    '''
    def __init__(self, Nparticles, r0, dr1,dr2, n, avoidance=None, rho=1.,dims=[1000,1000,1000],resolution=None,
                 unit='rad*pixel',margin=10,maxr=None,PF=0, mu=np.nan,smoothingfac=None, incoherent=False,\
                 sticky=[None, None, None],xrdata=None, mask=None,mubg=None,**kwargs):
        ''' Initialize the simulator.'''
        # based on MRO, this will call Nmer3
        super(Annuli3Simulator,self).__init__(Nparticles, r0, dr1, dr2, n,rho=rho,dims=dims, \
                        resolution=resolution, unit=unit,avoidance=avoidance, mask=mask, xrdata=xrdata, mubg=mubg,**kwargs)
        self.addOne = self.addAnnuli


class ScatProperties:
    ''' The scattering properties of the system.
        deltarhoe - electron density (in e-/Angs^3)
        dpix - pixel dimensions (same units as Rdet, say meters)
        Rdet - sample - detector distance (same units as dpix, say meters)
        center - detector center
        flux - flux (in cts/s/m^2)
        wv - wavelength
        absorption - absorption factor, some number between 0 and 1
        exposuretime - the exposure time
        dims - dimensions of array (if there is an array of pixels)
    '''
    def __init__(self, deltarhoe, dpix, Rdet, flux, wv, absorption, exposuretime, dims=[1,1], center=[0,0]):
        if(absorption < 0 or absorption > 1):
            print("Warning, absorption factor is not between zero and 1")
        #dsigma domega
        self.deltarhoe = deltarhoe # in e-/nm**3
        self.dpix = dpix
        self.Rdet = Rdet
        self.flux = flux
        self.absorption = absorption
        self.wv = wv
        self.exposuretime = exposuretime
        self.center = center
        self.dims = dims

        self.dsdo = constants.R0**2*deltarhoe**2
        self.domega = (self.dpix/self.Rdet)**2

        self.qperpixel = self.calc_q()
        self.fluxfactor = self.calcfluxfactor()

    def calc_q(self):
        ''' calc_q is a common tool taken from tools/optics.py
            calc_q(L,a,wv) - calculate the q value for length L, transverse
                distance a and wavelength wv.
            Use this to calculate the speckle size

            L - sample to detector distance (mm)
            a - pixel size transverse length from beam direction (mm)
            wv - wavelength
            Units of L and a should match and resultant q is in inverse units of wv.
           '''
        return calc_q(self.Rdet,self.dpix,self.wv)

    def getresolution(self):
        ''' Get the resolution necessary in Angstroms per pixel an image of
            dimension N to have a resolution in the Fourier domain of N
            of qperpixel.'''
        return 2*np.pi/self.dims[0]/self.qperpixel

    def calcfluxfactor(self):
        ''' Return the flux factor, which is meant to be multiplied times the V^2 S(q)
            to obtain the scattering counts expected.
            dsdo * flux * delta omega * (absorption factor)
        '''
        fluxfactor = self.dsdo*self.flux*self.domega*self.absorption*self.exposuretime
        return fluxfactor
