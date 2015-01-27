import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import math
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

fig=plt.figure(figsize=(16,9), dpi=400)
ax = fig.add_subplot(111)
ax.set_xlim(min(xs),max(xs))
ax.set_ylim(min(ys),max(ys))
ax.scatter(xs, ys, s=mag, color = 'r', marker = '.', linewidth = "1")
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
    
