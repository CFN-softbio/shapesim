import numpy as np
cimport numpy as np
# for the cython decorators
cimport cython

from libc.math cimport sin, cos

ctypedef fused anytype:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _rotate(np.ndarray[anytype, ndim=1, mode="c"] img1,
           np.ndarray[anytype, ndim=1, mode="c"] img2,
           double theta, double cenx, double ceny, int dimx, int dimy):
    cdef double cth, sth
    cdef int xr, yr
    cdef int i, j

    cth = cos(theta);
    sth = sin(theta);
    for i in range(dimx):
        for j in range(dimy):
            xr = (int)((i-cenx)*cth - (j-ceny)*sth + cenx + .5)
            yr = (int)((i-cenx)*sth + (j-ceny)*cth + ceny + .5)
            # rotate by rotating source coordinates, not destination
            # the latter would lead to missing points
            if((xr >= 0) and (xr < dimx) and (yr >= 0) and (yr < dimy)):
                img2[i + dimx*j] = img1[xr + dimx*yr]

def rotate(np.ndarray[anytype, ndim=2, mode="c"] img1,
           np.ndarray[anytype, ndim=2, mode="c"] img2,
           double theta, double cenx, double ceny):
    cdef int dimx = img1.shape[0]
    cdef int dimy = img1.shape[0]
    cdef np.ndarray[anytype, ndim=1, mode="c"] imga = img1.ravel()
    cdef np.ndarray[anytype, ndim=1, mode="c"] imgb = img2.ravel()
    _rotate(imga, imgb, theta, cenx, ceny, dimx, dimy);


def zero_elems(np.ndarray[np.int_t, ndim=2, mode="c"] arr, int nopixels):
    '''
    Quickly zero an array at pixels indices, meant to use with rotate
        and meant to be fast.
    '''
    cdef int i;
    for i in range(nopixels):
        arr[i] = 0
