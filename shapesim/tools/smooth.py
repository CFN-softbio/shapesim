from scipy.ndimage.filters import gaussian_filter
import numpy as np

def smooth2Dgauss(img, mask=None, sigma=30):
    ''' Smooth an image in 2D according to the mask.
        sigma: the sigma to smooth by
    '''
    if mask is None:
        mask = np.ones_like(img)
    if sigma is None:
        return img
    imgf = np.copy(img)
    imgg = gaussian_filter(img*mask, sigma)
    imgmsk = gaussian_filter(mask*1., sigma)
    imgmskrat = gaussian_filter(mask*0.+1, sigma)
    w = np.where(imgmsk > 0)
    imgf[w] = imgg[w]/imgmsk[w]*imgmskrat[w]
    return imgf
