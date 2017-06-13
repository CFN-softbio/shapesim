# Born Approximation SAXS simulator
This is a crude set of code that allows one to simulate the scattering
of finite sized shapes using the Born approximation at small angle.


This is meant to serve as a quick method to produce projections of
various shapes as well as their scattering. Each may be manipulated in
3D space. The actual resultant projected images and scattering are not
computed until the `project()` method is called.

Two main approximations are made here : 
1. Born approximation : the sample is sufficiently far away from the
   detector
2. Small angle approximation : the detector is assumed to sample
   Cartesian space.

Note, the second condition is only satisfied for 2D planar samples. When
the displacements in the longitudinal direction are large, you will see
effects from qz even though |qz| is much smaller than |qx| or |qy| (ref
[2]).

```
[1] Angular Correlations:
Lhermitte, J. R., Tian, C., Stein, A., Rahman, A., Zhang, Y., Wiegart,
L., Fluerasu, A., Gang, O. & Yager, K. G. (2017). J. Appl. Crystallogr.
50. DOI : 10.1107/S160057671700394610.1107/S1600576717003946

[2] X-ray Amplification:
(accepted) Coherent Amplification of X-ray Scattering from
Meso-structures Lhermitte, J. R., Stein, A., Tian, C., Zhang, Y.,
Wiegart, L., Fluerasu, A., Gang, O. and Yager, K. G. 
```

Note : the actual code used to generate the simulated scattering is very
simple (abs(fft(img))^2). This code was more meant to be fast and
flexible, leading to more easily reproducible Monte-Carlo simulations.
It is recommended you first try simulating your own patterns yourself by
creating a density map and just running the transformation mentioned
here.


Example:
```python
In [1]: from shapesim.shapes import HexLattice3Spheres

In [2]: shp = HexLattice3Spheres(3, 12, 6)

# project the image before you may see it
In [3]: shp.project()

# the projected image is shp.img
In [4]: shp.img.real.shape
Out[4]: (1000, 1000)

# the scattering is shp.fimg2
In [5]: shp.fimg2.real.shape
Out[5]: (1000, 1000)

# rotate about z axis (out of plane) by .1 radians
In [6]: shp.rotz(.1)

# need to project again
In [7]: shp.project()
```

More examples to be added later...
