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
def anim():
    f = mlab.gcf()
    x=1
    while 1:
        f.scene.camera.azimuth(.5)
        f.scene.render()
        #mlab.savefig("/home/christopher/code/Physics/image{}.png".format(x), figure=f, size=(1920,1080))
        x += 1
        yield
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



points = np.array(zip(xs,ys,zs))
ckdtree = space.cKDTree(points,3)

neighbors = []

for point in points:
    neighbors.append(len(ckdtree.query_ball_point(point,1)))

print(max(neighbors))
    
for x in range(len(neighbors)):
    neighbors[x]=float(neighbors[x])/float(max(neighbors))

mlab.points3d(np.array(xs),np.array(ys),np.array(zs),neighbors)#scale_factor="0.5")
#mlab.show()

a = anim() # Starts the animation.
mlab.show()
#fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xs, ys, zs, c='r', marker = 'o')


#plt.show()
