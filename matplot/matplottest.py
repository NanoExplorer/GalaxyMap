import matplotlib

matplotlib.use("TkAgg")

import matplotlib.backends.backend_pdf as pdfback
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#from mayavi import mlab
print("Loading Coordinates...")

xs = []#list of x coordinates of galaxies. The coordinates of galaxy zero are (xs[0],ys[0],zs[0])
ys = []
zs = []

with open("./BoxOfGalaxies.csv", "r") as boxfile:
    for line in boxfile:
        if line[0]!="#":
            try:
                row = line.split(',')
                xs.append(float(row[14]))
                ys.append(float(row[15]))
                zs.append(float(row[16]))
            except ValueError:
                pass

#mlab.points3d(np.array(xs),np.array(ys),np.array(zs))
#mlab.show()
print("Normalizing...")
center = ((max(xs)-min(xs))/2,(max(ys)-min(ys))/2,(max(zs)-min(zs))/2,)
xs = [x - center[0] for x in xs]
ys = [y - center[1] for y in ys]
zs = [z - center[2] for z in zs]
print("Generating Spherical Coords...")
phis = [math.acos(z/(math.sqrt(x**2+y**2+z**2))) - (math.pi/2) for x,y,z in zip(xs,ys,zs)]
#The minus pi/2 is because acos returns in the range [0,pi] when the map expects elevations from [-pi/2, pi/2]
thetas = [math.atan2(y,x) for x,y in zip(xs,ys)]
#phis and thetas are the phi and theta spherical coordinates accd. to my calculus book (except for elevations)
rho = [math.sqrt(x**2+y**2+z**2) for x,y,z in zip(xs,ys,zs)]
mag = 1#[500/(x**2+y**2+z**2) for x,y,z in zip(xs,ys,zs)]


print("Generating plots...")
fig=plt.figure(figsize=(16,9), dpi=400)
ax = fig.add_subplot(111, projection='hammer')
ax.scatter(thetas, phis, s=mag, color = 'r', marker = '.', linewidth = "1")
ax.set_title('Angular Distribution of Galaxies')
plt.xlabel('azimuth')
plt.ylabel('elevation')
#ax2 = fig.add_subplot(212)
#numbins = 50
#ax2.hist(rho,numbins,color='g',alpha=0.8)

fig2=plt.figure(figsize=(15,5), dpi=400)
hs = fig2.add_subplot(131)
hs2= fig2.add_subplot(132)
hs3= fig2.add_subplot(133)
numbins = 50
hs.hist (xs,numbins,color='g',alpha=0.8)
hs2.hist(ys,numbins,color='g',alpha=0.8)
hs3.hist(zs,numbins,color='g',alpha=0.8)
hs.set_title ('Distribution of Galaxies by X position')
hs2.set_title('Distribution of Galaxies by Y position')
hs3.set_title('Distribution of Galaxies by Z position')
    
#ax = fig.add_subplot(131, projection='hammer')
#ax.scatter(thetas, phis, c='r', marker = 'o')

#ax2 = fig.add_subplot(132, projection='3d')
#ax2.scatter(xs, ys, zs, c='r', marker = 'o')

#ax3 = fig.add_subplot(133)
#ax3.scatter(phis, thetas, c = 'r', marker = 'o')

#plt.show()
print("Saving plots...")
with pdfback.PdfPages('out.pdf') as pdf:    
    pdf.savefig(fig)
    pdf.savefig(fig2)

print("Done!")
