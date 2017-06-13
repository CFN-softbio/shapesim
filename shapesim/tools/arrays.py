import numpy as np

def mkarray3D(nx, ny, nz, vecx, vecy, vecz, vecs, dd=None, vec0=None):
    ''' Make 3D array of particles and return the vectors.
        nx, ny, nz: number of particles in x,y,z
        vecx, vecy, vecz : basis vectors
        vecs : the vectors for all the subunits in a unit (a N by 3 array,
            where N can be one)
        dd : randomize the coordinates (+/- d)
    '''
    if(dd is None):
        dd = 0
    nx = np.array(nx)
    ny = np.array(ny)
    nz = np.array(nz)
    if(len(nx.shape) == 0):
        nx0 = nx/2.
        nx = np.array([0, nx])
    else:
        nx0 = 0

    if(len(ny.shape) == 0):
        ny0 = ny/2.
        ny = np.array([0, ny])
    else:
        ny0 = 0

    if(len(nz.shape) == 0):
        nz0 = nz/2.
        nz = np.array([0, nz])
    else:
        nz0 = 0

    vecstot = np.array([])
    vecstot.shape=(0,3)
    if(vec0 is None):
        vec0 = nx0*vecx + ny0*vecy + nz0*vecz
    for i in range(nx[0],nx[1]):
        for j in range(ny[0], ny[1]):
            for k in range(nz[0], nz[1]):
                for vec in vecs:
                    rnd = (np.random.random(3)-.5)*dd
                    vecstot = np.vstack((vecstot, vec + vecx*i + vecx*rnd[0] + vecy*j + vecy*rnd[1] + vecz*k + vecz*rnd[2] - vec0))
    return vecstot

def mkarray2D(nx, ny, vecx, vecy, vecs, dd=None, vec0=None):
    ''' Make 2D array of particles and return the vectors.
        nx, ny: number of particles in x,y
        vecx, vecy : basis vectors
        vecs : the vectors for all the subunits in a unit (a N by 2 array,
            where N can be one)
        dd : randomize the coordinates (+/- d)
    '''
    vecx = np.array(vecx)
    vecy = np.array(vecy)
    vecs = np.array(vecs)

    if(dd is None):
        dd = 0
    nx = np.array(nx)
    ny = np.array(ny)

    if(len(nx.shape) == 0):
        nx0 = nx/2.
        nx = np.array([0, nx])
    else:
        nx0 = 0

    if(len(ny.shape) == 0):
        ny0 = ny/2.
        ny = np.array([0, ny])
    else:
        ny0 = 0

    vecstot = np.array([])
    vecstot.shape=(0,2)
    if(vec0 is None):
        vec0 = nx0*vecx + ny0*vecy 
    for i in range(nx[0],nx[1]):
        for j in range(ny[0], ny[1]):
            for vec in vecs:
                rnd = (np.random.random(2)-.5)*dd
                vecstot = np.vstack((vecstot, vec + vecx*i + vecx*rnd[0] + vecy*j + vecy*rnd[1] - vec0))

    return vecstot

