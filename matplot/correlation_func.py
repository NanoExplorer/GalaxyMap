print("Loading...")
import matplotlib

matplotlib.use("TkAgg")
import time
import matplotlib.backends.backend_pdf as pdfback
import numpy as np
import scipy.spatial as space
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random
import itertools
NUM_PROCESSORS = 8

#from mayavi import mlab
def load(filename):
    print("Loading Coordinates...")

    xs = []#list of x coordinates of galaxies. The coordinates of galaxy zero are (xs[0],ys[0],zs[0])
    ys = []
    zs = []

    with open(filename, "r") as boxfile:
        for line in boxfile:
            if line[0]!="#":
                try:
                    row = line.split(',')
                    xs.append(float(row[14]))
                    ys.append(float(row[15]))
                    zs.append(float(row[16]))
                except ValueError:
                    pass
    return (xs,ys,zs)

"""
def d_p_est(r,dr,actual_kd,random_kd):
    #from http://ned.ipac.caltech.edu/level5/March04/Jones/Jones5_2.html
    #Nrd = N so the factor Nrd/N = 1 and will be left out.
    #DD(r) = average number of pairs
    #DR(r) = average num pairs between random and actual
    lower = r-(dr/2)
    assert(lower >= 0)
    upper = r+(dr/2)

    DD = actual_kd.count_neighbors(actual_kd,np.array([lower,upper]))
    DR = actual_kd.count_neighbors(random_kd,np.array([lower,upper]))
    print('.',end="",flush=True)
    return ((DD[1]-DD[0])/(DR[1]-DR[0]))-1
"""
def hamest(min_value,max_value,step,actual_list,random_list):
    """
    Notes: step = 2*dx
    min_value - dx >> 0 (or else we find a distance that is slightly greater than zero
    and we end up with a zero galaxy count and a divide by zero error)
    
    if min value is 1 and max value is 6 and step is one, we should generate a range
    [1,2,3,4,5,6(!!!!)]
    then subtract step/2 for
    [.5,1.5,2.5,3.5,4.5,5.5]
    so that the range generated ends up being in the middle of each pair (list[0],list[1]) or (list[1],list[2]) etc
    of values in the resultant list.
    """
    actual_kd = space.cKDTree(actual_list,3)
    random_kd = space.cKDTree(random_list,3)
    num_elements = int((max_value-min_value)/step)+1+1 #one because range is not inclusive,
                                                       #one because int rounds down (and we want an EXTRA element)
    intervals = [((x*step)+min_value)-(step/2) for x in range(num_elements)]
    #This one needs the extra value at the end for producing intervals from first-dx to last+dx
    
    xs = [(x*step)+min_value for x in range(num_elements-1)]
    #Here we DON'T want that extra element, so we       ^ subtract one from num_elements
    #this will be the desired list of x values
    
    check_list = np.array(intervals)
    print(intervals)
    lower = min(check_list)
    assert(lower >= 0)
    DDs = actual_kd.count_neighbors(actual_kd,check_list)
    DRs = actual_kd.count_neighbors(random_kd,check_list)
    RRs = random_kd.count_neighbors(random_kd,check_list)
    dxs = itertools.repeat(step/2) #error in each x value
    correlations = calculate_correlations(DDs,DRs,RRs)
    
    #return value looks like this: a list of tuples, (x value, x uncertainty, y value)
    return zip(xs,dxs,correlations)

def calculate_correlations(DDs,DRs,RRs):
    results = []
    for index in range(len(DDs)-1):
        DDr = DDs[index+1]-DDs[index]
        DRr = DRs[index+1]-DRs[index]
        RRr = RRs[index+1]-RRs[index]
        results.append((DDr*RRr)/(DRr**2)-1)
    return results

    
def makeplot(xs,ys,title,xl,yl):
    fig = plt.figure(figsize=(4,3),dpi=100)
    ax = fig.add_subplot(111)
    ax.loglog(xs,ys,'o')
    ax.set_title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

def unwrap(zvals):
    xs = []
    ys = []
    for tup in zvals:
        xs.append(tup[0])
        ys.append(tup[2])
    return (xs,ys)
        
def main():
    master_bins = []
    master_corrs = []
    xs, ys, zs = load("./BoxOfGalaxies.csv")
    cubic_min = min(min(xs),min(ys),min(zs))
    cubic_max = max(max(xs),max(ys),max(zs))
    num_galax = len(xs)
    assert(len(xs) == len(ys) == len(zs))
    actual_galaxies = np.array(list(zip(xs,ys,zs)))
    print("    Generating random data set and corresponding tree...")
    random_galaxies = np.random.uniform(cubic_min,cubic_max,(num_galax,3))
    print("    Computing correlation function...")
    start = time.time()
    correlation_func_of_r = hamest(1,20,.1,actual_galaxies,random_galaxies)
    print("That took {:.2f} seconds".format(time.time()-start))
    print("Complete.")
    xs,ys = unwrap(correlation_func_of_r)
    makeplot(xs,ys,"Correlation function of distance r","Distance(Mpc/h)","correlation")

def writecsv(xslist,yslist):
    assert(len(xslist)==len(yslist))
    with open("./out2.csv",'w') as csv:
        for row in range(len(xslist[0])):
            line = ""
            for cell in range(len(xslist)):
                line = line + str(xslist[cell][row]) + ',' + str(yslist[cell][row])+ ','
            line = line + '\n'
            csv.write(line)
if __name__ == "__main__":
    main()
"""
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
"""
