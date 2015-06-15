import common
from scipy.spatial import cKDTree
import numpy as np
import math
from multiprocessing import Pool
import itertools
import pylab
import matplotlib.backends.backend_pdf as pdfback
#import pdb
from numpy.core.umath_tests import inner1d #Note: this function has no documentation and could easily be deprecated.
#if that happens, you can always use the syntax (a*b).sum(axis=1), which is ever so slightly slower and much more
#ugly.
#Alternate definition:
# def inner1d(a,b):
#     return (a*b).sum(axis=1)

def length(a):
    #Deprecated in favor of using numpy arrays for everything
    return math.sqrt(a[0]**2+a[1]**2+a[2]**2)

def psiOneNumerator(rv1, rv2, cosdTheta):
    """
    Calculates \psi 1's numerator as defined in the Gorski paper
    This is one iteration. So call this function a billion times then add up all the results.
    """
    return rv1*rv2*cosdTheta

def psiOneDenominator(cosdTheta):
    return (cosdTheta)**2

def psiTwoNumerator(rv1,rv2,costheta1,costheta2):
    return rv1*rv2*costheta1*costheta2

def psiTwoDenominator(costheta1,costheta2,cosdTheta):
    return costheta1*costheta2*cosdTheta

#@profile
def main(args):
    np.seterr(divide='ignore',invalid='ignore')
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
    #Get setup information from the settings file
    settings = common.getdict(args.settings)
    numpoints = settings["numpoints"]
    dr =        settings["dr"]
    min_r =     settings["min_r"]
    outfolder = settings["output_data_folder"]
    outfile   = settings["output_file_name"]
    step_type = settings["step_type"]
    rawInFile = settings["input_file"]
    pool = Pool()
    if settings["many"]:
        #If there are lots of files, set them up accordingly.
        inFileList = [rawInFile.format(x+settings['offset']) for x in range(settings["num_files"])]
    else:
        inFileList = [rawInFile]

    xs,intervals = common.genBins(min_r,numpoints,dr,step_type)

    plots = pool.starmap(singlePlot,zip(inFileList,
                                        itertools.repeat(intervals)))
    
    for n,plot in enumerate(plots):
        psione = [x for x in plot[0]]
        psitwo = [y for y in plot[1]]
        a = [z for z in plot[2]]
        common.writedict(outfolder+outfile.format(n+settings['offset'])+'_rawdata.json',{'psione':psione,
                                                                                         'psitwo':psitwo,
                                                                                         'a':a,
                                                                                         'xs':xs})
        #End for loop
    stats(args)


#@profile
def singlePlot(infile,intervals):
    #Load the survey
    galaxies = common.loadData(infile, dataType = "CF2")

    #Make an array of just the x,y,z coordinate and radial component of peculiar velocity (v)
    galaxyXYZV = np.array([(a.x,a.y,a.z,a.v) for a in galaxies])

    #Put just the galaxy positions into one array
    positions = galaxyXYZV[:,0:3] # [(x,y,z),...]

    kd = cKDTree(positions)
    #Get the galaxy pairs in each bin
    pairs = kd.query_pairs(max(intervals))
    #list(pool.starmap(kd_query,zip(itertools.repeat(positions),
             #                                      interval_shells)))
    #Calculate the actual correlations
    psiOne,psiTwo,a = correlation(pairs,galaxyXYZV,intervals)

    #Divide by 10^4 as per the convention in Borgani
    psione = psiOne / 10**4
    psitwo = psiTwo / 10**4

    #Write the data to a file for use by the 'stats' method
    return (psione,psitwo,a)

#@profile
def correlation(pairs,galaxies,intervals):
    # galaxies = [(galaxies[a],galaxies[b]) for a,b in interval_shell] 
    galaxyPairs = np.array(list(pairs)) #This line takes more time than all the others in this method combined...
    lGalaxies = galaxies[galaxyPairs[:,0]]
    rGalaxies = galaxies[galaxyPairs[:,1]]

    #"Galaxy 1 VelocitieS"
    g1vs = lGalaxies[:,3]
    g2vs = rGalaxies[:,3]
    g1pos = lGalaxies[:,0:3]
    g2pos = rGalaxies[:,0:3]

    #Distances from galaxy to center
    g1dist = np.linalg.norm(g1pos,axis=1)
    g2dist = np.linalg.norm(g2pos,axis=1)

    #Normalized galaxy position vectors
    #The extra [:,None] is there to make the denominator axis correct. See
    #http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    g1norm = g1pos/g1dist[:,None] 
    g2norm = g2pos/g2dist[:,None]
    
    distBetweenG1G2 = np.linalg.norm(g2pos-g1pos,axis=1)
    
    #r is a unit vector pointing from g2 to g1
    r = (g2pos-g1pos) / distBetweenG1G2[:,None]

    cosdTheta = inner1d(g1norm,g2norm)
    cosTheta1 = inner1d(r,g1norm)
    cosTheta2 = inner1d(r,g2norm)

    #The 'ind' stands for individual
    indPsiOneNum = psiOneNumerator(g1vs,g2vs,cosdTheta)
    indPsiOneDen = psiOneDenominator(cosdTheta)
    indPsiTwoNum = psiTwoNumerator(g1vs,g2vs,cosTheta1,cosTheta2)
    indPsiTwoDen = psiTwoDenominator(cosTheta1,cosTheta2,cosdTheta)
    indANum = aNumerator(cosdTheta,g1dist,g2dist,distBetweenG1G2)
    indADen = aDenominator(cosdTheta,distBetweenG1G2)

    #The numpy histogram function returns a tuple of (stuff we want, the bins)
    #Since we already know the bins, we throw them out by taking the [0] element of the tuple.
    psiOneNum = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiOneNum)[0]
    psiOneDen = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiOneDen)[0]
    psiTwoNum = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiTwoNum)[0]
    psiTwoDen = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiTwoDen)[0]
    aNum      = np.histogram(distBetweenG1G2,bins = intervals,weights = indANum)[0]
    aDen      = np.histogram(distBetweenG1G2,bins = intervals,weights = indADen)[0]
    
    psione = psiOneNum/psiOneDen
    psitwo = psiTwoNum/psiTwoDen
    a = aNum/aDen
    return (psione,psitwo,a)

# Old abandoned function. Does the same thing as above, but less efficiently
# def single_correlation(galaxy1,galaxy2):
#     rv1 = galaxy1.v
#     rv2 = galaxy2.v
#     g1pos =np.array((galaxy1.x,galaxy1.y,galaxy1.z))
#     g2pos =np.array((galaxy2.x,galaxy2.y,galaxy2.z))
#     g1dist = length(g1pos)
#     g2dist = length(g2pos)
#     cosdTheta = np.dot(g1pos/g1dist,g2pos/g2dist)
#     distBetweenG1G2 =length(g2pos-g1pos)
#     r = (g2pos-g1pos) / distBetweenG1G2
#     costheta1 = np.inner(r,g1pos/g1dist)
#     costheta2 = np.inner(r,g2pos/g2dist)
#     psiOneNum = psiOneNumerator(rv1,rv2,cosdTheta)
#     psiOneDen = psiOneDenominator(cosdTheta)
#     psiTwoNum = psiTwoNumerator(rv1,rv2,costheta1,costheta2)
#     psiTwoDen = psiTwoDenominator(costheta1,costheta2,cosdTheta)
#     aNum = aNumerator(cosdTheta,g1dist,g2dist,distBetweenG1G2)
#     aDen = aDenominator(cosdTheta,distBetweenG1G2)
#     return (psiOneNum,psiOneDen,psiTwoNum,psiTwoDen,aNum,aDen)

def aNumerator(cosdTheta,g1d,g2d,r):
    return (g1d*g2d*(cosdTheta-1)+(r**2)*cosdTheta)*cosdTheta

def aDenominator(cosdTheta,r):
    return (cosdTheta**2)*(r**2)
    
#@profile
def kd_query(positions,interval):
    """Get all pairs of galaxies at distances inside the interval. 
    interval is a tuple of (upper,lower) bounds for the distance bin.
    """
    kd = cKDTree(positions)
    upper = kd.query_pairs(interval[0])
    lower = kd.query_pairs(interval[1])
    upper.difference_update(lower) #Get rid of pairs at separations less than the lower bound
    # Use this instead of one - two because of time complexity.
    #see: https://wiki.python.org/moin/TimeComplexity
    
    return upper

#@profile
def stats(args):
    """Make plots of the data output by the main function"""
    #Get settings
    settings = common.getdict(args.settings)
    outfolder = settings["output_data_folder"]
    outfile   = settings["output_file_name"]
    rawInFile = settings["input_file"]
    
    if settings["many"]:
        inFileList = [outfolder+outfile.format(x+settings['offset'])+'_rawdata.json' for x in range(settings["num_files"])]
    else:
        inFileList = [outfolder+outfile+'_rawdata.json']

    for n,infile in enumerate(inFileList):
        data = common.getdict(infile)
        xs = data['xs']
        a = data['a']
        psione = data['psione']
        psitwo = data['psitwo']

        fig = pylab.figure()
        pylab.plot(xs,a,'-',label="$\cal A$")
        pylab.title("Moment of the selection function")
        pylab.ylabel("$\cal A$")
        pylab.xlabel("Distance, Mpc/h")
        #pylab.yscale('log')
        #pylab.xscale('log')
        pylab.axis((0,31,.62,.815))
        pylab.legend()

        fig2 = pylab.figure()
        pylab.plot(xs,psione,'-',label="$\psi_1$")
        pylab.plot(xs,psitwo,'k--',label="$\psi_2$")
        pylab.title("Velocity correlation function")
        pylab.xlabel("Distance, Mpc/h")
        pylab.ylabel("Correlation, $10^4 (km/s)^2$")
        pylab.axis((0,31,0,32))
        pylab.legend()

        with pdfback.PdfPages(outfolder+outfile.format(n+settings['offset'])) as pdf:
            #pdf.savefig(fig)
            pdf.savefig(fig2)
        pylab.close('all')

class FakeArgs:
    def __init__(self, filename):
        self.settings = filename

if __name__ == "__main__":
    settingsFile = input("Input settings file name: ")
    print("Input function.")
    print("1: only computations.")
    print("2: only plotting.")
    print("3: compute correlations then plot.")
    choice = input("Your choice? ")
    arrrghs = FakeArgs(settingsFile)
    if choice == '1' or choice == '3':
        print("computing...")
        main(arrrghs)
    if choice == '2' or choice == '3':
        print('plotting...')
        stats(arrrghs)

    


