# a tile shape wrapper for 2d tile shapes.
# maybe this could be made into an object eventually
from .matrices import rotmat3D
import numpy as np

def mktilevecs(d,tilebin,vecx=None, vecy=None,vecz=None,phi=None, sigma=None):
    ''' Make 3d vectors of 2D tiles (z component will be zero)'''
    if(len(tilebin.shape) != 2):
        print("Not a 2D tile, exiting")
        return -1

    if(vecx is None):
        vecx = np.array([1,0,0])
    if(vecy is None):
        vecy = np.array([0,1,0])
    if(vecz is None):
        vecz = np.array([0,0,1])
    if(phi is None):
        phi = 0.
    if sigma is None:
        sigma = 0.

    w = np.where(tilebin != 0)
    vecs = np.zeros((len(w[0]),3))
    vecs[:,0] = w[1]*d
    vecs[:,1] = w[0]*d
    vecs += (np.random.random((len(w[0]),3))-.5)*d*sigma*2
    center = np.average(vecs,axis=0)
    vecs -= center[np.newaxis,:]
    rot = rotmat3D(phi)
    vecs = np.tensordot(vecs,rot,axes=[[1],[0]])
    return vecs
