import numpy as np
from mayavi import mlab


x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j] # [-5,

def scalars(x,y,z):
    return x * x * 0.5 + y * y + z * z * 2.0


obj = mlab.contour3d(x,y,z,scalars, transparent=True)
#testpt = mlab.points3d(np.array([10]),np.array([10]),np.array([10]))
mlab.show()
