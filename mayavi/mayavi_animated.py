import numpy as np
import scipy.spatial as space
import time
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from mayavi import mlab
import vtk

#xs = [1.3,4.3,5.6]
#ys = [1.4,6.5,3.4]
#zs = [7.5,3.5,8.54]
@mlab.animate(delay=10)
def anim(f1,f2):
    x=0
    while x<(360*2):
        f1.scene.camera.azimuth(.5)
        f1.scene.render()
        f2.scene.camera.azimuth(.5)
        f2.scene.render()
        mlab.savefig("/home/christopher/code/Physics/galaxy{}.png".format(x), figure=f1, size=(1920/2,1080))
        mlab.savefig("/home/christopher/code/Physics/random{}.png".format(x), figure=f2, size=(1920/2,1080))
        x += 1
        yield
    exit()
xs = []
ys = []
zs = []

with open("/home/christopher/code/Physics/millenium/BoxOfGalaxies.csv", "r") as boxfile:
    for line in boxfile:
        if line[0]!="#":
            try:
                row = line.split(',')
                xs.append(float(row[14]))
                ys.append(float(row[15]))
                zs.append(float(row[16]))
            except ValueError:
                pass



mlab.figure(bgcolor=(0,0,0),size=(1000,1200))
gal_plot = mlab.gcf()
mlab.points3d(np.array(xs),np.array(ys),np.array(zs),scale_factor="0.5",color=(1,0,0),opacity=0.5)
mlab.figure(bgcolor=(0,0,0),size=(1000,1200))
ran_plot = mlab.gcf()
print("Reticulating Splines...")
random_points = np.random.uniform(0,62.5,(3,8795))


print("Overlaying Grid on Bezier Curves...")
mlab.points3d(random_points[0],
                            random_points[1],
                            random_points[2],
                            scale_factor="0.5",
                            color=(0,1,1),
                            opacity=0.5)

print("Displaying isotropically generated imagery!")


a = anim(gal_plot,ran_plot) # Starts the animation.
mlab.show()
