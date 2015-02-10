import numpy as np
import scipy.spatial as space
from mayavi import mlab
print("Reticulating Splines...")
random_points = np.random.uniform(0,62.5,(3,8795))


print("Overlaying Grid on Bezier Curves...")
fig = mlab.figure(bgcolor=(0,0,0))
mlab.points3d(random_points[0],
              random_points[1],
              random_points[2],
              scale_factor="0.5",
              color=(0,1,1),
              opacity=0.5)
              
print("Displaying isotropically generated imagery!")
mlab.show()
