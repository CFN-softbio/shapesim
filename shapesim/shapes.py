''' These are for 3D projected shapes.
    In this case, the shapes can be manipulated in 3 dimensions.
    They come with a few routines to rotate them etc.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .tools.transforms import rotation_matrix
from .tools.matrices import rotmat3D, rotEuler
from .tools.arrays import mkarray3D
from .tools.rotate import rotate
from .tools.tiles import mktilevecs


class Shape3:
    ''' 3D shape class.
        It is made up of types and units. Types are generators of shapes and
        units are the positions of these shapes.
        This makes it possible to quickly generate a structure (such as a crystal)
        of the same shape, while allowing the possibility of constructing a hybrid
        crystal of different shapes (i.e. spheres of different sizes/density).
    '''
    def __init__(self,dims=[1000,1000,1000],resolution=None,unit='rad*pixel',poisson=False,xrdata=None):
        ''' Initialize object.
            dimensions are the dimensions of the object.
            Note: the object is not initialized in memory until
                asked for.
            The default is [1000,1000,1000] but this could change in the future.
            resolution - the resolution (in m) per pixel
                (resolution = 1 sets units to be pixels)
        xrdata : an xray data object. Must contain the following:
            xrdata.dims
            xrdata.qperpixel (assumes Ewald curvature negligible)
            xrdata.unit (default to inv angs)
            xrdata.x0
            xrdata.y0

        '''
        if resolution is None:
            resolution = 1.


        if xrdata is not None:
            # need to compute dims res etc
            self.N, self.IMGQBOX = self.getNQBOX(xrdata.dims,xrdata.x0,xrdata.y0)
            self.dims = [self.N, self.N, self.N]
            self.resolution = self.getresolution(xrdata.qperpixel, self.N)
            self.x0 = xrdata.x0
            self.y0 = xrdata.y0
            if hasattr(xrdata, 'unit'):
                self.unit = xrdata.unit
            else:
                self.unit = "Angs"
        else:
            self.N = None
            self.IMGQBOX = None
            self.resolution = resolution
            self.unit = unit
            self.dims = dims

        self.typebboxes = None
        self.set_size(self.dims, self.resolution, self.unit)


        # The positions and type of each unit
        self.vecs = np.array([],dtype=float).reshape(0,3)
        self.types = np.array([],dtype=int)

        # just so that inheriting classes can keep track of elements they add
        # (clear units shifts things around)
        self.ids = np.array([],dtype=int)

        # number of subunits
        self.notypes = 0

        # the data representing each subunit
        self.typefns = []
        self.typeparams = []
        self.typebboxes = []

        # temporary stuff
        self.curtype = -1

    def getNQBOX(self, DIMS,XCEN,YCEN,N=None):
        ''' Get the square dimensions and the subselection of the box
            for the simulation necessary.
            Y are rows and X are columns
        '''
        N = np.max(DIMS)*2
        XCEN_sim, YCEN_sim = N//2, N//2

        # get delta of lower left corners
        # TODO : Need to double check center not off by 1
        # the corner location of data in simulation
        CORNERX_data, CORNERY_data = -XCEN, -YCEN
        #CORNERX_sim, CORNERY_sim = N//4, N//4
        # then reshift to origin of simulation

        IMGQBOX = np.array([CORNERX_data + XCEN_sim, CORNERX_data + XCEN_sim + DIMS[1],
                            CORNERY_data + YCEN_sim, CORNERY_data + YCEN_sim + DIMS[0]])

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

    def getescat(self):
        ''' Get the electric field scattering.
            Note : this is the complex field
        '''
        if self.IMGQBOX is not None:
            x0,x1,y0,y1 = self.IMGQBOX.astype(int)
            return self.fimg[y0:y1,x0:x1]
        else:
            return self.fimg

    def set_size(self, dims, resolution=1., unit='rad*pixel'):
        ''' Set size (or resize) to dims and resolution etc.'''
        self.resolution = resolution
        self.unit = unit

        if self.typebboxes is not None:
            self.typebboxes = (np.array(self.typebboxes)/resolution)

        self.dims = dims
        self.dim2D = np.array(dims[:2])

        # some stuff specifying images
        self.img = None
        self.fft2 = None

        # temporary stuff
        self.typeimg = None

        #ideally we should have a qperpixel in x y and z
        # but let's assume cube units (square pixels etc)
        N = dims[0]
        self.qperpixel = 2*np.pi/N/resolution
        self.xperpixel = resolution

    def set_resolution(self,resolution):
        if resolution is not None:
            self.set_size(self.dims, resolution, self.unit)

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        self.vecs = np.array([],dtype=float).reshape(0,3)
        self.types = np.array([],dtype=int)

    def addtype(self, shapefn, shapeparams, bboxdims=None):
        ''' Add another shape type. Returns the number of this type.'''
        self.notypes += 1
        self.typefns.append(shapefn)
        self.typeparams.append(shapeparams)
        self.typebboxes.append(bboxdims)
    
    def addunits(self, vecs, typeno=None, avoidance = None):
        ''' Add a unit
            vecs : 3D position vectors (can also just be one vector)
                - can also be another Shape3 with vectors
            typeno : the type number of the unit (0 based indexing)
            Note : if no type specified, it will find last defined type.
                Also note that this is not the type of the last added unit,
                this is the last type that was defined.
            types are 0 based indexed. notypes is always max(types)+1
            2. This is slow. It is assumed that adding vecs is not something
                done often. It will recreate a numpy array for the vectors each time.
                If you need to speed this up, use a buffer (double memory usage
                every time it fills up)
            avoidance : if not None, make sure element has nothing in a sphere of this element in the way
                if there is conflict with at least one element, do not place any of the elements
                if avoidance is negative, then place all vectors
            return number of elements placed or -1 for cancellation

            About projection : The function is converted into pixels at the projection step.
                So all vectors are in the units specified until the projection.
        '''
        if isinstance(vecs, Shape3):
            ''' Can take another shape, if so use its vecs.'''
            vecs = vecs.vecs
        vecs = np.array(vecs)
        if(self.notypes == 0):
            raise ValueError("No types have been defined yet.")
        if(typeno is None):
            typeno = self.notypes-1
        if(typeno > self.notypes - 1):
            raise ValueError("The type specified does not exist")
        if(len(vecs.shape) == 1):
            vecs = np.array([vecs])
        elif(len(vecs.shape) == 2):
            if(vecs.shape[1] != 3):
                raise ValueError("The vector is not a list of 3 elements or a list of lists of 3 elements")
        vecs = np.array(vecs)
        typenos = np.tile(typeno,vecs.shape[0])

        if avoidance is not None:
            #check overlap
            if avoidance < 0:
                checkall = True
            else:
                checkall = False
            overlaps = self.overlapping(self.vecs,vecs,np.abs(avoidance),checkall=checkall)
            if(overlaps == -1):
                return -1
            elif(overlaps != 0 and isinstance(overlaps,list)):
                # this means we checked but returned the overlapping vecs, ignore them
                vecs = vecs[overlaps,:]

        # the slow piece
        N0 = self.vecs.shape[0]
        self.ids = np.concatenate((self.ids,np.arange(N0,N0+vecs.shape[0]+1)))
        self.vecs = np.concatenate((self.vecs,vecs))
        self.types = np.concatenate((self.types,typenos))
        # reorder in terms of types (will help make projection faster)
        vecsorder = np.argsort(self.types)
        self.vecs = self.vecs[vecsorder]
        self.types = self.types[vecsorder]
        self.ids = self.ids[vecsorder]


    def overlapping(self, vecs1, vecs2, radius,checkall=False):
        ''' Check if any of vecs2 are overlapping with any of vecs1.'''
        overlaps = list()
        for i, vec1 in enumerate(vecs1):
            for j, vec2 in enumerate(vecs2):
                vdiff = vec2-vec1
                if(vdiff[0]**2 + vdiff[1]**2 + vdiff[2]**2 < radius**2):
                    #there is overlap
                    if(checkall is False):
                        return -1
                    else:
                        overlaps.append(j)    
        if checkall:
            return overlaps
        else:
            return 1

    def project(self):
        ''' Project the units onto a 2D image. The convention
            is to project onto the x-y plane.'''
        if self.img is None:
            self.img = np.zeros((self.dims[1], self.dims[0]),dtype=complex)
        else:
            self.clearimg(self.img)
        curtype = -1
        for vec, typeno in zip(self.vecs, self.types):
            if typeno != curtype:
                # first clear old shape
                self.switchtype(typeno)
            # project vector onto z, round, project current type stored
            # removed + .5 in position (not sure why it was there?)
            self.projecttype((np.array(vec[:2])/self.resolution + self.dim2D/2.).astype(int))
        self.fimg = np.fft.fftshift(np.fft.fft2(self.img))
        self.fimg2 = np.abs(self.fimg)**2

    def switchtype(self, typeno):
        curtype = self.curtype
        if(self.typeimg is None):
            self.typeimg = np.zeros((self.dims[1],self.dims[0]),dtype=complex)
        if curtype >= 0:
            self.clearimg(self.typeimg, bboxdims=self.typebboxes[curtype])
        self.curtype = typeno
        # make a new type
        self.gentype(typeno)

    def transform3D(self, tranmat):
        ''' Transform the 3D vectors according the transformation 
            matrix.'''
        #tvecs = np.array(vecs)*0;
        tranmat = np.array(tranmat)
        self.vecs = np.tensordot(self.vecs,tranmat,axes=(1,1))

    def center(self):
        ''' Center object. Shift vectors by center.'''
        COM = np.average(self.vecs,axis=0)
        self.translate3D(-COM)

    def rotz(self,phi):
        ''' Rotate about the z axis (in projection plane).'''
        rotmat = rotmat3D(phi,axis=3)
        self.transform3D(rotmat)

    def roty(self,phi):
        ''' Rotate about the x axis (in projection plane).'''
        rotmat = rotmat3D(phi,axis=2)
        self.transform3D(rotmat)

    def rotx(self,phi):
        ''' Rotate about the y axis (in projection plane).'''
        rotmat = rotmat3D(phi,axis=1)
        self.transform3D(rotmat)

    def translate3D(self, vec):
        ''' Translate the 3D vectors according the translation vector.'''
        #tvecs = np.array(vecs)*0;
        vec = np.array(vec)
        self.vecs += vec[np.newaxis,:]
        
    def gentype(self, typeno):
        ''' Make the temporary type specified by the typeno.'''
        curbbox = np.array(self.typebboxes[typeno]).astype(int)
        # project sphere onto the image
        self.typefns[typeno](self.typeimg,*(self.typeparams[typeno]),bboxdims=curbbox,resolution=self.resolution)
       
    def projecttype(self, r): 
        ''' Project the current type in the class to the image at the position specified.
            bbox indexing is left biased: [-bd[0]//2, (bd[0]-1)//2]
        '''
        bd = np.array(self.typebboxes[self.curtype]).astype(int)
        # xleft, yleft, xright, yright
        bboxtype = [0, 0, bd[0]-1, bd[1]-1]
        bboximg = [r[0] - bd[0]//2, r[1] - bd[1]//2, r[0] + (bd[0]-1)//2, r[1] + (bd[1]-1)//2]
        #bounds check (need to speed this up later)

        # x check, check min is in bounds
        if(bboximg[0] < 0):
            bboxtype[0] -= bboximg[0]
            bboximg[0] = 0
            
        if(bboximg[0] >= self.img.shape[1]):
            # out of bounds so don't plot
            return

        # x check, check max is in bounds
        if(bboximg[2] < 0):
            # out of bounds so don't plot
            return
            
        if(bboximg[2] >= self.img.shape[1]):
            bboxtype[2] -= (bboximg[2] - (self.img.shape[1]-1))
            bboximg[2] = self.img.shape[1] - 1

        # y check, check min is in bounds
        if(bboximg[1] < 0):
            bboxtype[1] -= bboximg[1]
            bboximg[1] = 0
            
        if(bboximg[1] >= self.img.shape[0]):
            # out of bounds so don't plot
            return

        # y check, check max is in bounds
        if(bboximg[3] < 0):
            # out of bounds so don't plot
            return
            
        if(bboximg[3] >= self.img.shape[0]):
            bboxtype[3] -= (bboximg[3] - (self.img.shape[0]-1))
            bboximg[3] = self.img.shape[0] - 1

        #during check, it's possible that the type box bounds are empty, check:
        if(bboxtype[0] >= bboxtype[2] or bboxtype[1] >= bboxtype[3]):
            return
        
        if(bboximg[0] != bboximg[2] and bboximg[1] != bboximg[3]):
            self.img[bboximg[1]:bboximg[3]+1,bboximg[0]:bboximg[2]+1] += self.typeimg[bboxtype[1]:bboxtype[3]+1,bboxtype[0]:bboxtype[2]+1]


    def plotbbox(self, vecno):
        ''' Plot an image with the bounding box of the vector highlighted
            based on the vector number vecno
            Not implemented but the aim is to find which element a vector designates.
            Will project in 2D.
        '''
        pass

    def clearimg(self,img,bboxdims=None):
        ''' Clear the last projected shape.
            Note: setting bboxdims speeds things up.
            Set bboxdims = None to clear the whole image.
            This routine will eventually be written in cython.
        '''
        if(bboxdims is None):
            img *= 0
        else:
            img[:bboxdims[0], :bboxdims[1]] = 0

    def get2Dvecs(self):
        # return the 2D vecs for the hex lattice (useful for generating coordinates)
        return self.vecs[:,:2]

class Shape3Spheres(Shape3):
    ''' A Shape3 but where the generating function is always the same type.'''
    def __init__(self,radius,rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel',avoidance=None,bboxdims=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Shape3Spheres,self).__init__(dims=dims, resolution=resolution, unit=unit,**kwargs)
        if bboxdims is None:
            bboxcutoff = 2*(int(radius/resolution)+4)
            bboxdims = [bboxcutoff, bboxcutoff]
        self.rho = rho
        self.addtype(sphereprojfn,(radius,rho, 0., 0., 0.),bboxdims=bboxdims)

class Shape3Gausses(Shape3):
    ''' A Shape3 but where the generating function is always the same type.'''
    def __init__(self,radius,rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel',avoidance=None,bboxdims=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Shape3Gausses,self).__init__(dims=dims, resolution=resolution, unit=unit,**kwargs)
        if bboxdims is None:
            # go 3 sigma
            bboxcutoff = 3*(int(radius/resolution)+4)
            bboxdims = [bboxcutoff, bboxcutoff]
        self.rho = rho
        self.addtype(gaussprojfn,(radius,rho, 0., 0., 0.),bboxdims=bboxdims)


class Shape3Superballs(Shape3):
    ''' A Shape3 but where the generating function is always the same type.
        Added a dummy param so it accepts same number of args as other functions
            (should fix this later).
    '''
    def __init__(self,radius, p, dummy=None, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel',bboxdims=None,avoidance=None, **kwargs):
        ''' like shape3 but new parameter is the radius of the superball.
            p - the superball factor p -> inf is a perfect square, p down to 2 is a rounded square
        '''
        if resolution is None:
            resolution = 1.
        super(Shape3Superballs,self).__init__(dims=dims, resolution=resolution, unit=unit,**kwargs)
        if bboxdims is None:
            bboxcutoff = 4*(int(radius/self.resolution)+1)
            bboxdims = [bboxcutoff, bboxcutoff]
            self.bboxdims = None
            print("bboxdims {} radius {} res {}".format(bboxdims,radius, resolution))
        else:
            # keep bbox dims fixed
            self.bboxdims = bboxdims
        self.ld = radius #effective distance from center
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.p = p
        self.radius = radius
        self.rho = rho
        self.addtype(superballprojfn,(radius,p,rho, 0., 0., 0.),bboxdims=bboxdims)
        self.avoidance = avoidance

    def addsuperball(self, vec0,r,p,phi=0.,rho=1.):
        ''' add a superball.
            vec0 : position
            r : 'radius' 
            p : superball factor
            phi : orientation
        '''
        if self.bboxdims is None:
            bboxcutoff = 4*(int(r/self.resolution)+1)
            bboxdims = [bboxcutoff, bboxcutoff]
        else:
            bboxdims = self.bboxdims
        self.addtype(superballprojfn,(r,p,rho,phi,0., 0.),bboxdims=bboxdims)
        avoidance = self.avoidance
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.vecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vec0)

class Shape3Ellipses(Shape3):
    ''' A Shape3 but where the generating function is always the same type.'''
    def __init__(self,r1, r2, alpha, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel',avoidance=None,bboxdims=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Shape3Ellipses,self).__init__(dims=dims, resolution=resolution, unit=unit,**kwargs)
        if bboxdims is None:
            bboxcutoff = 2*(int(np.maximum(r1,r2)/resolution)+4)
            bboxdims = [bboxcutoff, bboxcutoff]
        self.rho = rho
        self.addtype(ellipseprojfn,(r1, r2, rho, alpha, 0., 0.),bboxdims=bboxdims)

from shapes.arrays import mkarray3D
class Nmer3(Shape3Spheres):
    ''' An nmershape of spheres
        radius - sphere radius
        ld - lattice distance
        n - symmetry
     '''
    def __init__(self,radius,ld,n,rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', avoidance=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Nmer3,self).__init__(radius,rho=rho,dims=dims, resolution=resolution, unit=unit,**kwargs)
        self.ld = ld
        self.n = n
        self.Nmertypes = []
        self.Nmervecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(Nmer3, self).clearunits()
        self.Nmervecs = np.array([]).reshape(0,3)
        self.Nmertypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0])
        self.clearunits()
        self.addNmer(vec0)

    def countNmers(self):
        ''' return the count of Nmers.'''
        return self.Nmervecs.shape[0]

    def addNmer(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        avoidance = self.avoidance
        vec0 = np.array(vec0)
        if(np.abs(self.n) > 1):
            vecs = np.zeros((self.n,3))
            for i in np.arange(self.n):
                dn = 2*np.pi/float(self.n)*i
                vec = self.ld*np.array([np.cos(dn),np.sin(dn),0])
                vecs[i] += vec
        elif(np.abs(self.n) == 1):
            ''' Move sphere to make COM at center.'''
            vecs = np.array([0,0,0])
        if phi is not None:
            rotmat = rotmat3D(phi)
            vecs = np.tensordot(vecs,rotmat,axes=(1,1))

        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.Nmervecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.Nmervecs = np.concatenate((np.array(self.Nmervecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()

class Nmer3Ellipses(Shape3Ellipses):
    ''' An nmershape of spheres
        radius - sphere radius
        ld - lattice distance
        n - symmetry
     '''
    def __init__(self,r1,r2, alpha, ld, n,rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', avoidance=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Nmer3Ellipses,self).__init__(r1,r2, alpha,rho=rho,dims=dims, resolution=resolution, unit=unit,**kwargs)
        self.ld = ld
        self.n = n
        self.Nmertypes = []
        self.Nmervecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(Nmer3Ellipses, self).clearunits()
        self.Nmervecs = np.array([]).reshape(0,3)
        self.Nmertypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0])
        self.clearunits()
        self.addNmer(vec0)

    def countNmers(self):
        ''' return the count of Nmers.'''
        return self.Nmervecs.shape[0]

    def addNmer(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        avoidance = self.avoidance
        vec0 = np.array(vec0)
        if(np.abs(self.n) > 1):
            vecs = np.zeros((self.n,3))
            for i in np.arange(self.n):
                dn = 2*np.pi/float(self.n)*i
                vec = self.ld*np.array([np.cos(dn),np.sin(dn),0])
                vecs[i] += vec
        elif(np.abs(self.n) == 1):
            ''' Move sphere to make COM at center.'''
            vecs = np.array([0,0,0])
        if phi is not None:
            rotmat = rotmat3D(phi)
            vecs = np.tensordot(vecs,rotmat,axes=(1,1))

        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.Nmervecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.Nmervecs = np.concatenate((np.array(self.Nmervecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()

class Annuli3(Shape3):
    ''' An nmershape of spheres
        radius - sphere radius
        r0 - initial radius
        dr1 - thickness of annulis
        dr2 - spacing between annuli
        n - number of rings
        height - max height of rings
        sigmar - error in ring width per ring
        sigmadr - error in ring position per ring
     '''
    def __init__(self,r0,dr1,dr2, n, height=None, sigmar=None,sigmadr=None, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', \
                avoidance=None,bboxdims=None,xrdata=None):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Annuli3,self).__init__(dims=dims, resolution=resolution, unit=unit,xrdata=xrdata)
        if bboxdims is None:
            bboxcutoff = min(2*int((r0+n*dr2+dr1/2.)/self.resolution)+1,self.dims[0])
            bboxdims = [bboxcutoff, bboxcutoff]
        if height is None:
            # then set height to thickness
            height = dr1

        self.rho = rho
        self.r0 = r0
        self.dr1 = dr1
        self.dr2 = dr2
        self.height = height
        self.n = n
        self.sigmar = sigmar
        self.sigmadr = sigmadr
        self.addtype(annuliprojfn,(r0,dr1,dr2,n, height, sigmar,sigmadr, rho, 0., 0., 0.),bboxdims=bboxdims)
        self.Annulitypes = []
        self.Annulivecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(Annuli3, self).clearunits()
        self.Annulivecs = np.array([]).reshape(0,3)
        self.Annulitypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0.])
        self.clearunits()
        self.addAnnuli(vec0)

    def countAnnuli(self):
        ''' return the count of Annuli.'''
        return self.Annulivecs.shape[0]

    def addAnnuli(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        vec0 = np.array(vec0) # force it to be numpy array
        avoidance = self.avoidance

        # nothing else to do, can't rotate about axis (symmetric)
        vecs = np.array([0,0,0.])
        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.Annulivecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.Annulivecs = np.concatenate((np.array(self.Annulivecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()

class Fresnel3(Shape3):
    ''' An nmershape of spheres
        radius - sphere radius
        rs - radii (central)
        drs - widths
        height - max height of rings
        sigmar - error in ring width per ring
        sigmadr - error in ring position per ring
     '''
    def __init__(self, rs, drs, height=None, sigmar=None,sigmadr=None, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', \
                avoidance=None,bboxdims=None,xrdata=None):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Fresnel3,self).__init__(dims=dims, resolution=resolution, unit=unit,xrdata=xrdata)
        if bboxdims is None:
            imax = np.argmax(rs)
            rmax = int((rs[imax] + drs[imax]*.5)/self.resolution) + 1# add 1 pixel
            bboxcutoff = min(2*rmax,self.dims[0])
            bboxdims = [bboxcutoff, bboxcutoff]
        if height is None:
            # then set height to thickness of first ring, not good assumption
            height = drs[0]

        rs = np.array(rs)
        drs = np.array(drs)

        self.rho = rho
        self.rs = rs
        self.drs = drs
        self.height = height
        self.sigmar = sigmar
        self.sigmadr = sigmadr
        self.addtype(fresnelprojfn,(rs,drs, height, sigmar,sigmadr, rho, 0., 0., 0.),bboxdims=bboxdims)
        self.Fresneltypes = []
        self.Fresnelvecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(Fresnel3, self).clearunits()
        self.Fresnelvecs = np.array([]).reshape(0,3)
        self.Fresneltypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0.])
        self.clearunits()
        self.addFresnel(vec0)

    def countAnnuli(self):
        ''' return the count of Annuli.'''
        return self.Fresnelvecs.shape[0]

    def addFresnel(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        vec0 = np.array(vec0) # force it to be numpy array
        avoidance = self.avoidance

        # nothing else to do, can't rotate about axis (symmetric)
        vecs = np.array([0,0,0.])
        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.Fresnelvecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.Fresnelvecs = np.concatenate((np.array(self.Fresnelvecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()



class FresnelShell3(Shape3):
    ''' An nmershape of spheres
        radius - sphere radius
        rs - radii (central)
        drs - widths
        height - max height of rings
        sigmar - error in ring width per ring
        sigmadr - error in ring position per ring
     '''
    def __init__(self, rs, drs, height=None, sigmar=None,sigmadr=None, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', \
                avoidance=None,bboxdims=None,xrdata=None):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(FresnelShell3,self).__init__(dims=dims, resolution=resolution, unit=unit,xrdata=xrdata)
        if bboxdims is None:
            imax = np.argmax(rs)
            rmax = int((rs[imax] + drs[imax]*.5)/self.resolution) + 1# add 1 pixel
            bboxcutoff = min(2*rmax,self.dims[0])
            bboxdims = [bboxcutoff, bboxcutoff]
        if height is None:
            # then set height to thickness of first ring, not good assumption
            height = drs[0]

        rs = np.array(rs)
        drs = np.array(drs)

        self.rho = rho
        self.rs = rs
        self.drs = drs
        self.height = height
        self.sigmar = sigmar
        self.sigmadr = sigmadr
        self.addtype(fresnelshellprojfn,(rs,drs, height, sigmar,sigmadr, rho, 0., 0., 0.),bboxdims=bboxdims)
        self.FresnelShelltypes = []
        self.FresnelShellvecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(FresnelShell3, self).clearunits()
        self.FresnelShellvecs = np.array([]).reshape(0,3)
        self.FresnelShelltypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0.])
        self.clearunits()
        self.addFresnelShell(vec0)

    def countAnnuli(self):
        ''' return the count of Annuli.'''
        return self.FresnelShellvecs.shape[0]

    def addFresnelShell(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        vec0 = np.array(vec0) # force it to be numpy array
        avoidance = self.avoidance

        # nothing else to do, can't rotate about axis (symmetric)
        vecs = np.array([0,0,0.])
        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.FresnelShellvecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.FresnelShellvecs = np.concatenate((np.array(self.FresnelShellvecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()

class CylinderShells3(Shape3):
    ''' An nmershape of cylindrical shells
        radius - sphere radius
        r0 - initial radius
        dr1 - thickness of annulis
        dr2 - spacing between annuli
        n - number of rings
     '''
    def __init__(self,r0,dr1,dr2, height, n, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', \
                avoidance=None,bboxdims=None,xrdata=None):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(CylinderShells3,self).__init__(dims=dims, resolution=resolution, unit=unit,xrdata=xrdata)
        if bboxdims is None:
            bboxcutoff = min(2*int((r0+n*dr2+dr1/2.)/self.resolution)+1,self.dims[0])
            bboxdims = [bboxcutoff, bboxcutoff]
        self.rho = rho
        self.r0 = r0
        self.dr1 = dr1
        self.dr2 = dr2
        self.n = n
        self.height = height
        self.addtype(cylindershellsprojfn,(r0,dr1,dr2,n, height, rho, 0., 0., 0.),bboxdims=bboxdims)
        self.CylinderShellstypes = []
        self.CylinderShellsvecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(CylinderShells3, self).clearunits()
        self.CylinderShellsvecs = np.array([]).reshape(0,3)
        self.CylinderShellstypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0.])
        self.clearunits()
        self.addCylinderShells(vec0)

    def countCylinderShells(self):
        ''' return the count of CylinderShells.'''
        return self.CylinderShellsvecs.shape[0]

    def addCylinderShells(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        vec0 = np.array(vec0) # force it to be numpy array
        avoidance = self.avoidance

        # nothing else to do, can't rotate about axis (symmetric)
        vecs = np.array([0,0,0.])
        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.CylinderShellsvecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.CylinderShellsvecs = np.concatenate((np.array(self.CylinderShellsvecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()

class Cylinder3(Shape3):
    ''' An cylinder
        currently long axis is along z direction, so it's basically a disc

        radius - cylinder radius
        height - cylinder height
     '''
    def __init__(self,radius,height, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', \
                avoidance=None,bboxdims=None,xrdata=None):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Cylinder3,self).__init__(dims=dims, resolution=resolution, unit=unit,xrdata=xrdata)
        if bboxdims is None:
            # TODO : get rotations of cylinder, take that into account
            bboxcutoff = 2*int(radius/self.resolution)+1
            bboxdims = [bboxcutoff, bboxcutoff]
        self.rho = rho
        self.radius = radius
        self.height = height
        self.addtype(cylinderprojfn,(radius,height, rho, 0., 0., 0.),bboxdims=bboxdims)
        self.Cylinderypes = []
        self.Cylindervecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(Cylinder3, self).clearunits()
        self.Cylindervecs = np.array([]).reshape(0,3)
        self.Cylindertypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0.])
        self.clearunits()
        self.addCylinder(vec0)

    def countCylinder(self):
        ''' return the count of Cylinder.'''
        return self.Cylindervecs.shape[0]

    def addCylinder(self, vec0,phi=0):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        vec0 = np.array(vec0) # force it to be numpy array
        avoidance = self.avoidance

        # nothing else to do, can't rotate about axis (symmetric)
        vecs = np.array([0,0,0.])
        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.Cylindervecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.Cylindervecs = np.concatenate((np.array(self.Cylindervecs,ndmin=2),vec0[np.newaxis,:]))

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()


class Tile3(Shape3Spheres):
    ''' An nmershape of spheres
        radius - sphere radius
        ld - lattice distance
        tilebin - the tile binary (a square array of zeros and ones)
     '''
    def __init__(self, radius, ld, tilebin, rho=1.,dims=[1000,1000,1000],resolution=None,unit='rad*pixel', avoidance=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(Tile3,self).__init__(radius,rho=rho,dims=dims, resolution=resolution, unit=unit,**kwargs)
        self.tilebins = np.array(tilebin)
        # make a list of tile bins with just 1 tile
        if self.tilebins.ndim ==2:
            self.tilebins = self.tilebins.reshape((1,self.tilebins.shape[0],self.tilebins.shape[1]))
        self.ld = ld
        self.Tiletypes = []
        self.Tilevecs = np.array([]).reshape(0,3)
        if avoidance is None:
            avoidance = -1
        self.avoidance = avoidance
        self.initialize()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(Tile3, self).clearunits()
        self.Tilevecs = np.array([]).reshape(0,3)
        self.Tiletypes = []

    def initialize(self,vec0=None):
        ''' initialize the shape'''
        if vec0 is None:
            vec0 = np.array([0,0,0])
        self.curtiletype = 0
        self.clearunits()
        self.addTile(vec0)

    def countNmers(self):
        ''' return the count of Nmers.'''
        return self.Tilevecs.shape[0]

    def addTile(self, vec0,phi=0,tiletype=None):
        ''' add an nmer at vec0.
            avoidance - the distance that counts as 'overlap'. Do not place sphere if
            it's in this overlap
            phi is the angle of rotation
        '''
        if tiletype is None:
            tiletype  = self.curtiletype
        avoidance = self.avoidance
        vec0 = np.array(vec0)
        vecs = mktilevecs(self.ld, self.tilebins[tiletype])
        if phi is not None:
            rotmat = rotmat3D(phi)
            vecs = np.tensordot(vecs,rotmat,axes=(1,1))

        vecs += vec0
        if avoidance > 0:
            # first check no overlap
            for vectmp in self.Tilevecs:
                dv = vectmp-vec0
                distance = np.sqrt(dv[0]**2 + dv[1]**2 + dv[2]**2)
                if(distance < avoidance):
                    # exit because it is overlapping
                    return -1
        self.addunits(vecs)
        self.Tilevecs = np.concatenate((np.array(self.Tilevecs,ndmin=2),vec0[np.newaxis,:]))

    def addTiles(self,vecs,phi=None):
        phi0 = phi
        for vec in vecs:
            if phi is None:
                phi0 = np.random.random()*2*np.pi
            self.addTile(vec,phi=phi0)

    def reset(self):
        ''' reset to original shape. '''
        self.initialize()

class HexLattice3Spheres(Shape3Spheres):
    ''' A hexagonal lattice of spheres.
        Could implement a superball later.
    '''
    def __init__(self, radius, ld, Narray, rho=1., dims = [1000,1000,1000], resolution=None,\
                 unit='rad*pixel',avoidance=None,**kwargs):
        ''' like shape3 but new parameter is the radius of the sphere.
            radius : sphere radius
            ld : nearest neighbor distance (lattice spacing is ld*sqrt(3)/2.)
        '''
        if resolution is None:
            resolution = 1.
        super(HexLattice3Spheres,self).__init__(radius,rho=rho,dims=dims, resolution=resolution, unit=unit,**kwargs)
        if avoidance is None:
            avoidance = -1

        self.avoidance = avoidance
        self.Narray = Narray
        self.ld = ld
        self.initialize()
        self.Hexvecs = np.array([]).reshape(0,3)

    def initialize(self):
        self.addHex()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(HexLattice3Spheres, self).clearunits()
        self.Hexvecs = np.array([]).reshape(0,3)
        self.Hextypes = []

    def getreff(self):
        ''' Get the effective radius reff of the sample.
            This doesn't take into account the size of the subelements.
            (The effective radius is centered on the sphere)
        '''
        return self.ld*np.sqrt(((self.Narray-1)*np.cos(60/57.3))**2 + ((self.Narray//2)*np.cos(30/57.3))**2)

    def reset(self):
        self.initialize()

    def addHex(self,vec0=None,phi=None):
        ''' Make coordinates for an NxN hex array.
            vec0 : center of hex array (default is COM of coordinates)
            phi : rotation angle
        '''
        Nrows = self.Narray
        distance = self.ld
        if vec0 is None:
            vec0 = np.array([0,0,0])
        else:
            vec0=np.array(vec0)
        # the square part
        vecs1 = [[0,0,0]]
        # the hex part
        vecs2 = [[.5*np.sqrt(3),.5,0]]
        
        vecx = np.array([0.,1.,0])
        vecy = np.array([np.sqrt(3), 0, 0])
        vecz = np.array([0,0,1.])
        hexvecs1 = mkarray3D([0,Nrows], [0,Nrows//2 + Nrows%2], 1, vecx, vecy, vecz, vecs1)
        hexvecs2 = mkarray3D([0,Nrows-Nrows%2],[0,Nrows//2], 1, vecx, vecy, vecz, vecs2)
        hexvecs = np.concatenate((hexvecs1,hexvecs2))
        center = np.average(hexvecs,axis=0)
        hexvecs -= center[np.newaxis,:]
        if(phi is not None):
            rotmat = rotmat3D(phi,axis=3)
            hexvecs = np.tensordot(hexvecs,rotmat,axes=(1,1))
        #rot = rotmat3D(phi)
        #hexvecs = np.tensordot(hexvecs,rot,axes=[[1],[0]])
        #hexvecs += center
        vecs = hexvecs*distance
        vecs += vec0
        self.addunits(vecs)



class CubicLattice3Spheres(Shape3Spheres):
    ''' A cubic lattice of spheres.
        Could implement a superball later.
    '''
    def __init__(self, radius, ld, Narray, rho=1., dims = [1000,1000,1000], resolution=None, unit='rad*pixel',avoidance=None):
        ''' like shape3 but new parameter is the radius of the sphere.'''
        if resolution is None:
            resolution = 1.
        super(CubicLattice3Spheres,self).__init__(radius,rho=rho,dims=dims, resolution=resolution, unit=unit)
        if avoidance is None:
            avoidance = -1

        self.avoidance = avoidance
        self.Narray = Narray
        self.ld = ld
        self.initialize()
        self.Cubevecs = np.array([]).reshape(0,3)

    def initialize(self):
        self.addCube()

    def clearunits(self):
        ''' Clear just the units (not the types).'''
        super(CubicLattice3Spheres, self).clearunits()
        self.Cubevecs = np.array([]).reshape(0,3)
        self.Cubetypes = []
        

    def reset(self):
        self.initialize()

    def addCube(self,vec0=None,phi=None):
        ''' Make coordinates for a cube
            vec0 : center of cube (default is COM of coordinates)
            phi : rotation angle
        '''
        Nrows = self.Narray
        distance = self.ld # lattice distance
        if vec0 is None:
            vec0 = np.array([0,0,0])
        else:
            vec0=np.array(vec0)

        #sub basis
        vecs = np.array([[0,0,0]])
        
        vecx = np.array([1., 0, 0])
        vecy = np.array([0., 1, 0])
        vecz = np.array([0., 0, 1])

        cubevecs = mkarray3D([0,Nrows], [0,Nrows], [0, Nrows], vecx, vecy, vecz, vecs)
        center = np.average(cubevecs,axis=0)
        cubevecs -= center[np.newaxis,:]

        if(phi is not None):
            rotmat = rotmat3D(phi,axis=3)
            cubevecs = np.tensordot(cubevecs,rotmat,axes=(1,1))

        vecs = cubevecs*distance
        vecs += vec0
        self.addunits(vecs)
 

# Some shape generating functions, will eventually be written in cython
def sphereprojfn(img, r, rho, alpha, beta, gamma, bboxdims=None,resolution=1.,off=None):
    ''' Draw the projection of a sphere.
        will be drawn in center of bounding box
        r - radius
        rho - density 
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.

        off - offset from pixel (use for subpixel resolution. only issue is shape must be now
            generated every time)
    '''
    if(bboxdims is None):
        bboxdims = img.shape
    # convert r from units to pixels
    rp = r/resolution
    rho = rho*resolution**2

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    if off is not None:
        rcen = np.array([bd[0]/2.+off[0], bd[1]/2.+off[1]])
    else:
        rcen = np.array([bd[0]/2., bd[1]/2.])
    

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    rhor = 2*rho*r
    img[:bd[1],:bd[0]] = rhor*np.sqrt(np.maximum(0,(1 - ((X-rcen[0])**2 + (Y-rcen[1])**2)/rp**2)))

def gaussprojfn(img, r, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of a sphere.
        will be drawn in center of bounding box
        r - radius (treated as the sigma)
        rho - density 
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
    '''
    if(bboxdims is None):
        bboxdims = img.shape
    # convert r from units to pixels
    r = r/resolution
    rho = rho*resolution**2

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    rhor = 2*rho*r
    img[:bd[1],:bd[0]] = rhor*np.exp(-((X-rcen[0])**2 + (Y-rcen[1])**2)/2./r**2)

def annuliprojfn(img, r0, dr1, dr2, n, height, sigmar, sigmadr, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of an annulus.
        will be drawn in center of bounding box
        (all will be normalized from units to pixels)
        r0 - starting radius
        dr1 - thickness of annulus
        dr2 - spacing between annuli
        n - number to draw
        height - the height of annuli
        rho - density
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
    '''
    if(bboxdims is None):
        bboxdims = img.shape
    # convert r from units to pixels
    r0 = r0/resolution
    dr1 = dr1/resolution/2.# change from thickness to half thickness (radii)
    dr2 = dr2/resolution
    rho = rho*resolution**2

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    r = np.sqrt((X-rcen[0])**2 + (Y-rcen[1])**2)
    # height is already in units, no need to normalize
    rhor = rho*height
    img[:bd[1],:bd[0]] *= 0
    if sigmar is None:
        sigmar = 0.
    if sigmadr is None:
        sigmadr = 0.
    sigmar = sigmar/resolution
    sigmadr = sigmadr/resolution
    for i in range(n):
        drr = (np.random.random()-.5)*sigmar
        ddrr = (np.random.random()-.5)*sigmadr
        r00 = r0 + i*dr2 + ddrr
        img[:bd[1],:bd[0]] += rhor*np.sqrt(np.maximum(1-((r-r00)/(dr1+drr))**2,0))

def fresnelprojfn(img, rs, drs, height, sigmar, sigmadr, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of an annulus.
        will be drawn in center of bounding box
        (all will be normalized from units to pixels)
        r0 - starting radius
        dr1 - thickness of annulus
        dr2 - spacing between annuli
        n - number to draw
        height - the height of annuli
        rho - density
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
    '''
    if(bboxdims is None):
        bboxdims = img.shape
    # convert r from units to pixels
    rs = rs/resolution
    drs = drs/resolution
    n = len(rs)

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    r = np.sqrt((X-rcen[0])**2 + (Y-rcen[1])**2)
    # height is already in units, no need to normalize
    rhor = rho*height
    img[:bd[1],:bd[0]] *= 0
    if sigmar is None:
        sigmar = 0.
    if sigmadr is None:
        sigmadr = 0.
    sigmar = sigmar/resolution
    sigmadr = sigmadr/resolution
    for i in range(n):
        drr = (np.random.random()-.5)*sigmar
        ddrr = (np.random.random()-.5)*sigmadr
        r00 = rs[i]
        img[:bd[1],:bd[0]] += rhor*np.sqrt(np.maximum(1-((r-r00)/(drs[i]/2.+drr))**2,0))


def fresnelshellprojfn(img, rs, drs, height, sigmar, sigmadr, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of cylinder shells.

        This one assumes the annuli are shells as opposed to more sphere like

        will be drawn in center of bounding box
        rs, drs: radii and thicknesses
        n - number to draw
        rho - density
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
    '''
    if(bboxdims is None):
        bboxdims = img.shape
    # convert r from units to pixels
    rs = rs/resolution
    drs = drs/resolution
    n = len(rs)

    rho = rho*resolution**2
    L = height

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    r = np.sqrt((X-rcen[0])**2 + (Y-rcen[1])**2)
    rhor = L*rho
    img[:bd[1],:bd[0]] *= 0
    sigmar = None
    sigmadr = None
    if sigmar is None:
        sigmar = 0.
    if sigmadr is None:
        sigmadr = 0.
    sigmar = sigmar/resolution
    sigmadr = sigmadr/resolution
    for i in range(n):
        rtmp = rs[i]
        drtmp = drs[i]
        drr = (np.random.random()-.5)*sigmar
        ddrr = (np.random.random()-.5)*sigmadr
        r00 = rtmp + ddrr
        img[:bd[1],:bd[0]] += rhor*((1-((r-r00)/(drtmp*.5+drr))**2) > 0)


def cylindershellsprojfn(img, r0, dr1, dr2, n, height, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of cylinder shells.

        This one assumes the annuli are shells as opposed to more sphere like

        will be drawn in center of bounding box
        r0 - starting radius
        dr1 - thickness of annulus
        dr2 - spacing between annuli
        n - number to draw
        rho - density 
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
    '''
    if(bboxdims is None):
        bboxdims = img.shape
    # convert r from units to pixels
    r0 = r0/resolution
    dr1 = dr1/resolution/2.# change from thickness to half thickness (radii)
    dr2 = dr2/resolution
    rho = rho*resolution**2
    L = height

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0])
    y = np.arange(bd[1])
    X,Y = np.meshgrid(x,y)
    r = np.sqrt((X-rcen[0])**2 + (Y-rcen[1])**2)
    rhor = L*rho
    img[:bd[1],:bd[0]] *= 0
    sigmar = None
    sigmadr = None
    if sigmar is None:
        sigmar = 0.
    if sigmadr is None:
        sigmadr = 0.
    sigmar = sigmar/resolution
    sigmadr = sigmadr/resolution
    for i in range(n):
        drr = (np.random.random()-.5)*sigmar
        ddrr = (np.random.random()-.5)*sigmadr
        r00 = r0 + i*dr2 + ddrr
        img[:bd[1],:bd[0]] += rhor*((1-((r-r00)/(dr1+drr))**2) > 0)

def ellipseprojfn(img, ra, rb, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of an ellipse 
        will be drawn in center of bounding box
        ra - semimajor axis
        rb - semiminor axis
        rho - density 
        img - image to project to
        alpha - the tilt of the ellipse

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
        NOTE : need to fix density later (rhor= 2*rho*ra)
    '''
    if(bboxdims is None):
        bboxdims = img.shape

    # convert r from units to pixels
    ra = ra/resolution
    rb = rb/resolution
    rho = rho*resolution**2

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0]).astype(float)
    y = np.arange(bd[1]).astype(float)
    X,Y = np.meshgrid(x,y)

    if(alpha != 0.):
        Xp = np.copy(X)
        Yp = np.copy(Y)
        #print("rotating by alpha: {}".format(alpha))
        rotate(Xp,X,alpha,rcen[0],rcen[1])
        rotate(Yp,Y,alpha,rcen[0],rcen[1])
        #plt.clf();plt.imshow(X);plt.draw();plt.pause(.001);
    # NOTE : need to fix density later
    rhor = 2*rho*ra
    img[:bd[1],:bd[0]] = rhor*np.sqrt(np.maximum(0,(1 - ((X-rcen[0])**2/ra**2 + (Y-rcen[1])**2/rb**2))))
    sum1 = np.sum(img[:bd[1],:bd[0]])
    # smooth image (so rotate versions look better)
    img[:bd[1],:bd[0]] = gaussian_filter(img[:bd[1],:bd[0]],1)
    sum2 = np.sum(img[:bd[1],:bd[0]])
    img[:bd[1],:bd[0]] *= sum1/sum2

def cylinderprojfn(img, r, L, rho, alpha, beta, gamma, bboxdims=None,resolution=1.):
    ''' Draw the projection of a cylinder
        will be drawn in center of bounding box
        assumes that the length is along the rows and the radius goes along the columns

        r - radius
        L - length
        rho - density 
        img - image to project to
        alpha - the tilt of the ellipse

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
        NOTE : need to fix density later (rhor= 2*rho*ra)
    '''
    if(bboxdims is None):
        bboxdims = img.shape

    # convert r from units to pixels
    rp = r/resolution
    Lp = L/resolution
    rho = rho*resolution**2

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0])/2., (bd[1])/2.;

    x = np.arange(bd[0]).astype(float)
    y = np.arange(bd[1]).astype(float)
    X,Y = np.meshgrid(x,y)

    if(alpha != 0.):
        Xp = np.copy(X)
        Yp = np.copy(Y)
        #print("rotating by alpha: {}".format(alpha))
        rotate(Xp,X,alpha,rcen[0],rcen[1])
        rotate(Yp,Y,alpha,rcen[0],rcen[1])
        #plt.clf();plt.imshow(X);plt.draw();plt.pause(.001);
    # NOTE : need to fix density later
    rhor = L*rho
    img[:bd[1],:bd[0]] = rhor*((1 - ((X-rcen[0])**2/rp**2 + (Y-rcen[1])**2/rp**2)) > 0)
    sum1 = np.sum(img[:bd[1],:bd[0]])
    # smooth image (so rotate versions look better)
    #img[:bd[1],:bd[0]].real = gaussian_filter(img[:bd[1],:bd[0]].real,1)
    #img[:bd[1],:bd[0]].imag = gaussian_filter(img[:bd[1],:bd[0]].imag,1)
    sum2 = np.sum(img[:bd[1],:bd[0]])
    img[:bd[1],:bd[0]] *= sum1/sum2

# superball
def superballprojfn(img, r, p, rho, alpha, beta, gamma, bboxdims=None,resolution=1.,niter=None):
    ''' Draw the projection of a sphere.
        will be drawn in center of bounding box
        r - radius
        p - superball parameter (p=1 is sphere)
        rho - density 
        img - image to project to

        Bounding box details: for bounding boxes with even number of dimensions, the central 
            pixel selected is ambiguous. I choose a left biased system:
            [x0 - (dx-1)//2, x0 + dx//2]
        The motivation for having it centered on a pixel is to that it registers with the original image
            in order to copy it faster. This will lead to 1 pixel errors and bias. Need to think on this.
            Could change later.
        NOTE: Eulerian angles not implemented yet
    '''
    # number of iterations for sum
    if niter is None:
        niter = 1000
    if(bboxdims is None):
        bboxdims = img.shape

    # convert r from units to pixels
    r/=resolution
    rho = rho*resolution**2

    # just so i dont have to type too much
    bd = bboxdims
    # [x0, y0]
    rcen = (bd[0]-1)//2, (bd[1]-1)//2;

    x = np.arange(bd[0]).astype(float)
    y = np.arange(bd[1]).astype(float)
    X,Y = np.meshgrid(x,y)
    #import matplotlib.pyplot as plt
    if(alpha != 0.):
        Xp = np.copy(X)
        Yp = np.copy(Y)
        #print("rotating by alpha: {}".format(alpha))
        rotate(Xp,X,alpha,rcen[0],rcen[1])
        rotate(Yp,Y,alpha,rcen[0],rcen[1])
        #plt.clf();plt.imshow(X);plt.draw();plt.pause(.001);
    pp = 2*p
    # integrate along z, symmetric so just integrate one side
    zs = np.linspace(0, 2*r, niter)
    dz = zs[1]-zs[0]
    rhodz = dz*rho
    for z in zs:
        img[:bd[1],:bd[0]] += ((np.abs(z/r)**pp + (np.abs(X-rcen[0])/r)**pp + (np.abs(Y-rcen[1])/r)**pp) < 1)
    img[:bd[1],:bd[0]] *= 2*rhodz
