from numpy.linalg import inv as inversemat
from numpy.linalg import det as determinant
import numpy as np
from scipy.ndimage.interpolation import affine_transform as scipy_affine_transform


def affine_transform(IMG,transform,offset=[0.,0.],shift=[0.0]):
    '''  transform(IMG,transform,offset=[0.,0.])
        Transform an image using an affine transform transform about a fixed point offset
        Unlike scipy.ndimage, offset is the *fixed point*. This is *a lot* more 
        sensible for computation!!!
        Only works for rotations and linear scaling for now. Need to do a little extra lin 
            alg for non uniform scaling.
    '''
    itransform = inversemat(transform)
    offset = (np.array(offset) - np.array(offset).dot(itransform)/determinant(itransform))
    #offset -= np.array(shift).dot(itransform)#/determinant(itransform)
    img = scipy_affine_transform(IMG,transform,offset=offset)#,offset=[-N/2,-N/2])
    return img

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    #REF source: http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotate_euler(alpha, beta, gamma):
    ''' Rotate according to the Euler angles.'''
    A = rotation_matrix([0,0,1],alpha)
    B = rotation_matrix(np.tensordot([1,0,0],A,axes=(1,1)),beta)
    C = rotation_matrix(np.tensordot(np.tensordot([0,0,1],A,axes=(1,1)),B,axes=(1,1)),gamma)
    D = np.tensordot(A,B,axes=(1,1))
    D = np.tensordot(D,C,axes=(1,1))
    return D
