import numpy as np

def rotmat3D(phi,axis=3):
    '''3D rotation matrix about z axis.
        Counter-clockwise rotation is positive.
        axis: choose either:
            1 - x axis
            2 - y axis
            3 - z axis
    '''
    if axis == 3:
        return np.array([
            [np.cos(phi), np.sin(phi),0],
            [-np.sin(phi), np.cos(phi),0],
            [0, 0, 1]
        ])
    elif axis == 2:
        return np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)],
        ])
    elif axis == 1:
        return np.array([
            [1, 0, 0],
            [0, np.cos(phi), np.sin(phi)],
            [0, -np.sin(phi), np.cos(phi)],
        ])
    else:
        print("Error, not a good axis specified. Specified: {}".format(axis))

def rotEuler(alpha, beta, gamma):
    ''' 3D rotation about the Euler angles.
        hasn't been tested yet
    '''
    rmat1 = rotmat3D(alpha, axis=1)
    rmat2 = rotmat3D(alpha, axis=3)
    rmat3 = rotmat3D(alpha, axis=1)
    rmat = np.tensordot(rmat1,rmat2, axes=(1,0))
    rmat = np.tensordot(rmat, rmat3, axes=(1,0))
    return rmat
