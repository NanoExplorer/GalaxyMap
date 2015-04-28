import numpy as np
import scipy.spatial as space
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from mayavi import mlab

DIVS = 100

#xs = [1.3,4.3,5.6]
#ys = [1.4,6.5,3.4]
#zs = [7.5,3.5,8.54]

xs = []
ys = []
zs = []

with open("/home/christopher/code/Physics/millennium/BoxOfGalaxies.csv", "r") as boxfile:
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

#neighbors = []

#for point in points:
#    neighbors.append(len(ckdtree.query_ball_point(point,1)))

#print(max(neighbors))
    
#for x in range(len(neighbors)):
#    neighbors[x]=float(neighbors[x])/float(max(neighbors))

#mlab.points3d(np.array(xs),np.array(ys),np.array(zs),neighbors)#scale_factor="0.5")
mlab.points3d(np.array(xs),np.array(ys),np.array(zs),scale_factor="0.5")
x, y, z = np.mgrid[min(xs):max(xs):DIVS*1j, min(ys):max(ys):DIVS*1j, min(zs):max(zs):DIVS*1j]

#print "oX",x
#print "oY",y
#print "oZ",z

def densitysane(sanecoords):
    #x,y,z = sanecoords
    #sanecoords is a 3-tuple (x,y,z)
    #densityfunc returns rho(r)
    return len(ckdtree.query_ball_point(sanecoords,5))

def densityfunc(x,y,z):
    #print "fX",x
    #print "fY",y
    #print "fZ",z
    #print type(x)
    #print x.shape
    resultMatrix = [[[0 for i in range(DIVS)] for j in range(DIVS)] for k in range(DIVS)]
    #the "i" here stands for index
    for ix in range(DIVS):
        for iy in range(DIVS):
            for iz in range(DIVS):
                #actual spatial coordinates are going to have a 'c' after them
                xc = x[ix][iy][iz]
                yc = y[ix][iy][iz]
                zc = z[ix][iy][iz]
                resultMatrix[ix][iy][iz] = len(ckdtree.query_ball_point((xc,yc,zc),5))
    
    #print type(resultMatrix)    
    #print "result:",resultMatrix
    return np.ndarray(x.shape,buffer=np.array(resultMatrix))#len(ckdtree.query_ball_point((x,y,z),1))

obj=mlab.contour3d(x,y,z,densityfunc,contours=2,transparent=True)
mlab.show()
#fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xs, ys, zs, c='r', marker = 'o')


#plt.show()
