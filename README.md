# GPSphere

Simple example of Gaussian process regression on a sphere, using the _Gaia_ DR2 quasars. Code originally by Timothy Brandt (UCSB). 

Notebooks:
- `make_quasar_maps.ipynb`: Create the error-weighed-mean _Gaia_ DR2 quasar proper motions map along with associated uncertainties.
- `GPR_sphere.ipynb`: Simple example of Gaussian process regression on this map.

Code:
- `distances.py`: Code to calculate distance matrix for points on the sphere.
- `gaussianprocess.py`: Matérn covariance for GP repression and associated likelihood.

_This project was developed in part at the 2019 Santa Barbara Gaia Sprint, hosted by the Kavli Institute for Theoretical Physics (KITP) at the University of California, Santa Barbara. This research was supported in part at KITP by the Heising-Simons Foundation and the National Science Foundation under Grant No. NSF PHY-1748958._
