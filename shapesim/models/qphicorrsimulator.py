from ..tools.qphicorr import deltaphicorr, deltaphicorr_qphivals
from ..tools.qphiavg import qphiavg
from ..tools.convol import convol1d
import matplotlib.pyplot as plt
import numpy as np
from fit.compFit import compFit
import os.path

class dummyclass:
    pass

class QPHICorrSimulator:
    ''' Expects a simulator object with a routine to randomize
        This object is meant to simulate the intensity pattern from a random array of
        spheres and compute a q delta phi correlation.
        Note : you need to specify a row for this q delta phi correlation
        TODO : use a running average and running stdev to compute the full matrix
        ncutoff : the number of phis to ignore (if smoothing it helps to increase this
            to avoid spurious correlations from Poisson noise and the interparticle interference)
    '''
    def __init__(self, simulator, noqs, nophis, qrow, orient=False, mask=None, incoherent=False, r0=None,ncutoff=None):
        self.mask= mask
        self.orient = orient
        self.noqs = noqs
        self.nophis = nophis
        self.qrow = qrow
        self.simulator = simulator
        self.simulator.set_mask(mask)
        simulator.set_coherence(not incoherent)
        self.N = simulator.dims[0]
        if r0 is None:
            self.x0, self.y0 = self.N/2, self.N/2
        else:
            self.x0, self.y0 = r0
        self.simulator.project();
        # the cutoff function for the fitting
        if ncutoff is None:
            ncutoff = 3

        self.ncutoff=ncutoff


        if mask is not None:
            self.addmask(mask)
        self.initialize()
        self.compute()

        # first get model
        Nparticles = self.simulator.Nparticles
        muold = self.simulator.mu
        self.simulator.set_mu(None)
        self.simulator.Nparticles = 1
        self.simul(1)
        self.datafunc = np.copy(self.sqcphi[self.qrow])
        self.datafunc -= np.average(self.datafunc)
        self.datafunc = self.datafunc[self.ncutoff:self.nophis//2-self.ncutoff]
        self.simulator.Nparticles = Nparticles
        self.simulator.mu = muold

    def plot(self,plotwin=1,figlims=None,savefigpref=None,scatclims=None):
        simulator = self.simulator
        plt.figure(plotwin);
        plt.clf();
        plt.subplot(221);
        plt.imshow(simulator.img)
        if figlims is not None:
            plt.xlim(figlims[0],figlims[1])
            plt.ylim(figlims[3],figlims[2])
        plt.subplot(222);
        plt.imshow(simulator.fimg2)
        if scatclims is not None:
            plt.clim(scatclims)
        if figlims is not None:
            plt.xlim(figlims[0],figlims[1])
            plt.ylim(figlims[3],figlims[2])
        plt.subplot(223);
        plt.imshow(self.sqphi)
        if scatclims is not None:
            plt.clim(scatclims)
        plt.subplot(224);
        plt.imshow(self.sqcphi/self.sqcphi[:,0][:,np.newaxis])
        plt.clim(0,1);
        plt.draw();
        if savefigpref is None:
            plt.pause(0.001);
        else:
            plt.savefig("{}.png".format(savefigpref))

    def addmask(self, mask):
        ''' Add a mask to qpcsim.'''
        self.mask = mask.astype(float)
        self.sqphimask = qphiavg(mask,x0=self.x0,y0=self.y0,noqs=self.noqs,nophis=self.nophis,mask=self.mask)
        self.sqcphimask = convol1d(self.sqphimask,axis=-1)

    def compute(self):
        ''' Compute qphi correlations for an iteration.'''
        simulator = self.simulator
        self.sqphi = qphiavg(simulator.fimg2,x0=self.x0,y0=self.y0,noqs=self.noqs,nophis=self.nophis,mask=self.mask)
        self.sqcphi = convol1d(self.sqphi,axis=-1)
        if self.mask is None:
            self.sqcphi /= self.nophis
        else:
            self.sqcphi /= self.sqcphimask

    def randomize(self,orient=False):
        ''' randomize and project.'''
        self.simulator.randomize(orient=orient)

    def initialize(self):
        ''' set up initial variables.'''
        self.qvals, self.phivals = deltaphicorr_qphivals(self.simulator.fimg2,self.simulator.fimg2,
                                        noqs=self.noqs, nophis = self.nophis,x0=self.x0,y0=self.y0,mask = self.mask)
    
    def simul(self, niter,plot=None,PF=False,savefigpref=None,figlims=None,scatclims=[None, None]):
        ''' simulate the qphi corr calculations on this object.
            if plot is true, plot the instances
            Assumes square box for image
            PF : print flag
        '''
        self.initialize()
        self.sqcphitot = np.zeros((self.noqs, self.nophis),dtype=float)
        self.qrows = np.zeros((niter, self.nophis))
        simulator = self.simulator
        if figlims is None:
            xlims = [0,self.N]
            ylims = [self.N,0]
        else:
            xlims = figlims[0:2]
            ylims = [figlims[3], figlims[2]]
        if savefigpref is not None:
            plt.ioff();
        for i in range(niter):
            if(PF):
                print("iteration {:4d} of {:4d}".format(i+1,niter))
            self.randomize(orient=self.orient)
            self.compute()
            self.sqcphitot += self.sqcphi
            self.qrows[i] = self.sqcphi[self.qrow]
            if(plot is not None):
                self.plot(plotwin=plot)
            if savefigpref is not None:
                self.plot(plotwin=1, savefigpref="{}_simulation_{:04d}".format(savefigpref,i),figlims=figlims,scatclims=scatclims)
        self.sqcphitot /= niter
        self.muavg = np.sqrt(np.average(self.qrows))
        self.qrowavg = np.average(self.qrows,axis=0)
        self.qrowstd = np.std(self.qrows,axis=0)
        if savefigpref is not None:
            plt.ion();

    def fitmodel(self):
        ''' fit data and get Signal to noise.
            Fits one particle data to the actual data
        '''
        # fit the data function using compFit
        self.ffit = compFit(self.datafunc,Amin=0,Bmin=0)
        ydata = np.array([self.qrowavg[self.ncutoff:self.nophis//2-self.ncutoff]])
        self.ffit.fitrows(ydata)
        self.signal = self.ffit.getsignals()[0]
        self.noise = self.ffit.getnoises()[0]
        self.SN = self.ffit.getSNs()[0]

    def plotfit(self):
        self.ffit.plotrow(0)

    def getnoise(self):
        return self.noise

    def getsignal(self):
        return self.signal

    def getSN(self):
        return self.SN


class QPHICorrSimulatorVaried(QPHICorrSimulator):
    ''' 
        Takes a QPHICorrSimulator object and runs it varying a parameter.
        Parameters that may be varied (make any of these an array):
            - smoothingfac : smoothin kernel sdev for Gaussian kernel (in pixels)
            - Nparticles : number of particles 
            - maxr :  maximum radius of box ( in pixels)
            - avoidance : avoidance distance of elements
            - mu : count rate (defaults to None) Note this is abritrary,
                first check the count rate gives what you want
            - sticky: array of 3 vectors [stickyc, stickyr, stickytype] where
                    they are the chance of stickiness, effective radius, random number generator
                    respectively
                stickytype can be one of the following strings:
                    'expon' : exponential
                    'norm' :  normal (Gaussian)
                    'lognorm' : lognormal
                if stickytype is None, the rest are ignored

        extra flags:
            - incoherent : defaults to False
            - Niter : number of iterations to average qhi correlations over
            - Niter_simul : number of simulations to average over (signals and noises)


        name : the name of the object

        Initialize a simulation storage. Set a master file.
            From here data will be stored as :
                /index/dataname
            etc where index is the prefix of the script that ran the simulation 
                (presumably unique).
        '''
    def __init__(self, simulator, noqs, nophis, qrow,\
                    smoothingfac, Nparticles, maxr, avoidance, mu,\
                    incoherent, Niter, Niter_simul,\
                    mubg = None, name=None,sticky=[None,None,None],\
                    orient=False, mask=None,resolutions=None, lds=None):
        super(QPHICorrSimulatorVaried, self).__init__(simulator, noqs, nophis, qrow, orient=orient, \
                mask=mask, incoherent=incoherent)

        self.smoothingfac = smoothingfac
        self.Nparticles = Nparticles
        if resolutions is None:
            resolutions = 1
        self.resolutions  = resolutions
        self.maxr = maxr
        self.avoidance = avoidance
        self.mu = mu
        self.mubg = mubg
        self.sticky = sticky
        self.lds = lds

        self.incoherent = incoherent
        self.Niter = Niter
        self.Niter_simul = Niter_simul
        self.name = name
        self.mask = mask

        # part of the simulator, add to this object
        self.dims = self.simulator.dims
        self.radius = self.simulator.typeparams[0][0]
        self.simulatorname = self.simulator.__class__.__name__
        self.ld = self.simulator.ld
        if hasattr(self.simulator,'n'):
            self.sym = self.simulator.n
        if hasattr(self.simulator,'Narray'):
            self.Narray = self.simulator.Narray
        # set to the initial values (just to make reviewing easier)
        self._normalizeparams()
        self.checklens() # make sure params all have same length
        self.set_params(0)

    def _normalizeparams(self):
        ''' make sure params are 1d arrays (not scalar or higher order).'''
        self.smoothingfac = np.array(self.smoothingfac).reshape(-1)
        self.Niter = np.array(self.Niter).reshape(-1).astype(int)
        self.Niter_simul = np.array(self.Niter_simul).reshape(-1).astype(int)
        self.Nparticles = np.array(self.Nparticles).reshape(-1).astype(int)
        self.maxr = np.array(self.maxr).reshape(-1)
        self.avoidance = np.array(self.avoidance).reshape(-1)
        self.mu = np.array(self.mu).reshape(-1)
        self.mubg = np.array(self.mubg).reshape(-1)
        # resolution length scale of system
        self.resolutions = np.array(self.resolutions).reshape(-1)
        self.lds = np.array(self.lds).reshape(-1)
        # force sticky to be an array, it will default to strings
        self.sticky = np.array(self.sticky).reshape(-1)
        self.sticky = self.sticky.reshape((self.sticky.shape[0]//3,3))

    def _findmaxlen(self):
        maxlen = np.max(
            [   self.smoothingfac.shape[0],
                self.Nparticles.shape[0],
                self.maxr.shape[0],
                self.avoidance.shape[0],
                self.mu.shape[0],
                self.lds.shape[0],
                self.resolutions.shape[0],
                self.mubg.shape[0],
                self.sticky.shape[0],
                self.Niter.shape[0],
                self.Niter_simul.shape[0]
            ])
        return maxlen

    def checklens(self):
        ''' find feature with max length and make other features share this length'''
        self._normalizeparams()
        self.m = self._findmaxlen()
        self.checklen("smoothingfac", self.m)
        self.checklen("Nparticles", self.m,dtype=int)
        self.checklen("maxr", self.m)
        self.checklen("avoidance", self.m)
        self.checklen("mu", self.m)
        self.checklen("mubg", self.m)
        self.checklen("Niter", self.m,dtype=int)
        self.checklen("Niter_simul", self.m,dtype=int)
        self.checklen("sticky", self.m,isvec=True,dtype=str)
        self.checklen("resolutions", self.m)
        self.checklen("lds", self.m)

    def checklen(self, datstr, maxlen, isvec=False, dtype=None):
        ''' need to give string of variable name.
                isvec : element is actually a 1D vector (important for reshaping)
        '''
        dat = self.__dict__[datstr]
        if dat.shape[0] != maxlen:
            if dat.shape[0] != 1:
                # Error, arrays dont either have same length or equal 1
                raise ValueError
            else:
                if not isvec:
                    self.__dict__[datstr] = np.tile(dat, (maxlen))
                    if dtype is not None:
                        self.__dict__[datstr] = self.__dict__[datstr].astype(dtype)
                elif isvec:
                    self.__dict__[datstr] = np.tile(dat, (maxlen,1))
                    if dtype is not None:
                        self.__dict__[datstr] = self.__dict__[datstr].astype(dtype)

    def set_params(self, n):
        self.simulator.set_smoothingfac(self.smoothingfac[n])
        self.simulator.set_avoidance(self.avoidance[n])
        self.simulator.set_maxr(self.maxr[n])
        self.simulator.set_Nparticles(self.Nparticles[n])
        self.simulator.set_mu(self.mu[n])
        self.simulator.set_mubg(self.mubg[n])
        self.simulator.set_sticky(self.sticky[n])
        self.simulator.set_resolution(self.resolutions[n])
        if self.lds[n] is not None:
            self.simulator.set_ld(self.lds[n])
        

    def run(self):
        ''' Run the simulation.'''
        self.checklens()
        #Niter_simul = self.Niter_simul #the iterations for the simulation just one
        #Niter = self.Niter # the number of times to average the noises and signals calculated
        #self.qrows = np.zeros((Niter_simul, self.m, self.nophis))
        self.signals = np.zeros(self.m)
        self.noises = np.zeros(self.m)
        self.simulator.set_coherence(not self.incoherent)
        
        for i in range(self.m):
            # set all dep variables
            self.set_params(i)

            for j in range(self.Niter[i]):
                print("{},{} of {},{}".format(i,j,self.m, self.Niter[i]))
                # repeat Niter times, average the noises together
                self.simul(self.Niter_simul[i],PF=1)
            #qrows[:,i,:nophis] = qpcsim.qrows
                self.fitmodel()
                self.signals[i] += self.getsignal()
                self.noises[i] += self.getnoise()
            if i == 0:
                self.qrowfits = np.zeros((self.m, self.ffit.rows.shape[1]))
                self.qrowavgs = np.zeros((self.m, self.ffit.rows.shape[1]))
            self.qrowavgs[i] += self.ffit.rows[0]
            self.qrowfits[i] += self.ffit.bestfits[0]
    
            # the number of iterations can change versus i
            self.signals[i] /= self.Niter[i]
            self.noises[i] /= self.Niter[i]
            self.qrowavgs[i] /= self.Niter[i]
            self.qrowfits[i] /= self.Niter[i]
        
    def store(self, name , db, force=False, PF=True, **kwargs):
        ''' saves some default data from here as well as extra kwargs.

            index : some unique name (set force=True to overwrite)
            db : the database to store to (currently a dbstore object)
                What this will do is append the entry to some master file
                and pickle the data as an object.
            force : force the storage (overwrite)

            Once stored, the results should be loaded via db manager (tools/Store.py)
        '''
        dset = dummyclass()

        # simulator results
        dset.qrowavgs = self.qrowavgs
        dset.qrowfits = self.qrowfits
        dset.signals = self.signals
        dset.noises = self.noises

        # simulation tweaked parameters
        # these could be arrays
        dset.Nparticles = self.Nparticles
        dset.avoidance = self.avoidance
        dset.smoothingfac = self.smoothingfac
        dset.maxr = self.maxr
        dset.mu = self.mu
        dset.mubg = self.mubg
        # for stickyp, need to also save params so make strings:
        # string is "<name>: params: <params>"
        dset.sticky = list()
        for sticky in self.sticky:
            dset.sticky.append(sticky)
        #finally, these are constant throughout all simulations
        dset.incoherent = self.incoherent 
        if self.mask is not None:
            dset.mask = self.mask
        
        ncutoff = self.ncutoff

        # the flags
        dset.Niter = self.Niter
        dset.Niter_simul = self.Niter_simul
        dset.orient = self.orient

        # qphi correlation stuff
        dset.noqs = self.noqs
        dset.nophis = self.nophis
        dset.qrow = self.qrow
        dset.qvals = self.qvals
        # it's delta phi so subtract first row
        dset.phis = (self.phivals - self.phivals[0])[self.ncutoff:self.nophis//2-self.ncutoff]

        # part of the simulator
        dset.dims = self.simulator.dims
        dset.radius = self.radius
        dset.simulatorname = self.simulatorname
        dset.ld = self.ld
        if hasattr(self,'sym'):
            dset.sym = self.sym
        if hasattr(self,"Narray"):
            dset.Narray = self.Narray
        if self.mask is not None:
            dset.mask = self.mask
    
        # finally save extra arguments
        for key in kwargs:
            dset[key] = kwargs[key]
    
        db.add(name,dset,force=force,PF=PF)
        print("Results saved")
