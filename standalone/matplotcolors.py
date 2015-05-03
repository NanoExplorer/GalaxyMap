import matplotlib

matplotlib.use("TkAgg")

import matplotlib.backends.backend_pdf as pdfback
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
matplotlib.rc('font', size=30)
#from mayavi import mlab
print("Loading Coordinates...")

xs = []#list of x coordinates of galaxies. The coordinates of galaxy zero are (xs[0],ys[0],zs[0])
ys = []
zs = []
vxs = []
vys = []
vzs = []

def radialVelocity(x,y,z,vx,vy,vz):
    r = math.sqrt(x**2+y**2+z**2)
    ux = x/r
    uy = y/r
    uz = z/r
    vr = ux*vx + uy*vy + uz*vz
    return vr

def colorFunc(vr,maxVr):
    #if vr equals positive maxVr then all blue
    #if vr equals negative maxVr then all red
    colorCoeff = 1/2*math.sqrt(abs(vr/(maxVr)))
    if vr>0:
        return (colorCoeff+0.5,0,0.5-colorCoeff)
    else:
        return (0.5-colorCoeff,0,colorCoeff+0.5)

with open("./BoxOfGalaxies.csv", "r") as boxfile:
    for line in boxfile:
        if line[0]!="#":
            try:
                row = line.split(',')
                xs.append(float(row[14]))
                ys.append(float(row[15]))
                zs.append(float(row[16]))
                vxs.append(float(row[21]))
                vys.append(float(row[22]))
                vzs.append(float(row[23]))
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

vrs = [radialVelocity(x,y,z,vx,vy,vz) for x,y,z,vx,vy,vz in zip(xs,ys,zs,vxs,vys,vzs)]
maxVr = max([max(vrs),abs(min(vrs))])
colors = [colorFunc(vr,maxVr) for vr in vrs]

print("Generating plots...")
fig=plt.figure(figsize=(10,7.5), dpi=400)
gridSpec = matplotlib.gridspec.GridSpec(2, 1,height_ratios=[5,1])
gridSpec.update(wspace=0.05)
ax = plt.subplot(gridSpec[0], projection='hammer')
ax.scatter(thetas, phis, s=20, color = colors, marker = '.', linewidth = "0")
ax.set_title('Millennium Simulation Galaxies',y=1.08)
plt.xlabel('')
plt.ylabel('Elevation')
plt.grid(True)
cm = matplotlib.colors.ListedColormap([colorFunc(vr,1) for vr in np.arange(-1,1,0.1)])
cax = fig.add_subplot(gridSpec[1])
cb = matplotlib.colorbar.ColorbarBase(cax,cmap=cm, norm=matplotlib.colors.Normalize(vmin = -maxVr, vmax = maxVr),spacing='proportional',orientation='horizontal')
cb.set_label("Radial Velocity (km/s)")
plt.tick_params(axis='x',which='major',labelsize=20)
ax.set_xticklabels([])

# fig2=plt.figure(figsize=(15,5), dpi=400)
# hs = fig2.add_subplot(131)
# hs2= fig2.add_subplot(132)
# hs3= fig2.add_subplot(133)
# numbins = 50
# hs.hist (xs,numbins,color='g',alpha=0.8)
# hs2.hist(ys,numbins,color='g',alpha=0.8)
# hs3.hist(zs,numbins,color='g',alpha=0.8)
# hs.set_title ('Distribution of Galaxies by X position')
# hs2.set_title('Distribution of Galaxies by Y position')
# hs3.set_title('Distribution of Galaxies by Z position')
    
#ax = fig.add_subplot(131, projection='hammer')
#ax.scatter(thetas, phis, c='r', marker = 'o')

#ax2 = fig.add_subplot(132, projection='3d')
#ax2.scatter(xs, ys, zs, c='r', marker = 'o')

#ax3 = fig.add_subplot(133)
#ax3.scatter(phis, thetas, c = 'r', marker = 'o')



fig.subplots_adjust(bottom=.15,top=.85,left=.15,right=.93)
print("Saving plots...")
with pdfback.PdfPages('out.pdf') as pdf:    
    pdf.savefig(fig)
#    pdf.savefig(fig2)

print("Done!")
