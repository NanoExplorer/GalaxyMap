import matplotlib

matplotlib.use("TkAgg")

import matplotlib.backends.backend_pdf as pdfback
import numpy as np
from math import pi
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import common


#phis and thetas are the phi and theta spherical coordinates accd. to my calculus book (except for elevations)
#rho = [math.sqrt(x**2+y**2+z**2) for x,y,z in zip(xs,ys,zs)]
mag = 1#[500/(x**2+y**2+z**2) for x,y,z in zip(xs,ys,zs)]

galaxies = common.loadData("../matplot/Yuyu data/SimulCatalogue/CF2_group_simul.txt")
thetas = [galaxy.lon*(pi/180)-pi for galaxy in galaxies]
phis = [galaxy.lat*(pi/180) for galaxy in galaxies]

print("Generating plots...")
fig=plt.figure(figsize=(8,4.5), dpi=180)
ax = fig.add_subplot(111, projection='hammer')
ax.scatter(thetas, phis, s=mag, color = 'r', marker = '.', linewidth = "1")
ax.set_title('Angular Distribution of Galaxies')
plt.xlabel('azimuth')
plt.ylabel('elevation')
#ax2 = fig.add_subplot(212)
#numbins = 50
#ax2.hist(rho,numbins,color='g',alpha=0.8)


    
#ax = fig.add_subplot(131, projection='hammer')
#ax.scatter(thetas, phis, c='r', marker = 'o')

#ax2 = fig.add_subplot(132, projection='3d')
#ax2.scatter(xs, ys, zs, c='r', marker = 'o')

#ax3 = fig.add_subplot(133)
#ax3.scatter(phis, thetas, c = 'r', marker = 'o')

#plt.show()
print("Saving plots...")
with pdfback.PdfPages('SIM_Yuyu.pdf') as pdf:    
    pdf.savefig(fig)

print("Done!")
