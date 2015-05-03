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

print("Hello Numpy!")
invertedcoords = np.array([xs,ys,zs])
coords = transpose(invertedcoords)




















    
