import common
from scipy.spatial import cKDTree
import numpy as np
import math
from multiprocessing import Pool
import itertools
import pylab
import matplotlib.backends.backend_pdf as pdfback
from numpy.core.umath_tests import inner1d #Note: this function has no documentation and could easily be deprecated.
#if that happens, you can always use the syntax (a*b).sum(axis=1), which is ever so slightly slower and much more
#ugly.
#emergency abort function:
# def inner1d(a,b):
#     return (a*b).sum(axis=1)

def length(a):
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
    """
    Grab the CF2 file, chug it into cartesian (automatically done in common.py now!), plug into cKDTree, grab pairs
    plug information about pairs into psi functions, sum them, return values.
    """
    settings = common.getdict(args.settings)
    numpoints = settings["numpoints"]
    dr =        settings["dr"]
    min_r =     settings["min_r"]
    step_size = settings["step_size"]
    outfolder = settings["output_data_folder"]
    outfile   = settings["output_file_name"]
    step_type = settings["step_type"]
    rawInFile = settings["input_file"]
    pool = Pool()
    if settings["many"]:
        inFileList = [rawInFile.format(x+settings['offset']) for x in range(settings["num_files"])]
    else:
        inFileList = [rawInFile]
    for n,infile in enumerate(inFileList):
        galaxies = common.loadData(infile, dataType = "CF2")
        galaxyXYZV = np.array([(a.x,a.y,a.z,a.v) for a in galaxies])
        xs,intervals = common.intervals(min_r,step_size,numpoints,dr,step_type)
        positions = galaxyXYZV[:,0:3] # [(x,y,z),...]

        interval_shells = [(intervals[i+1],intervals[i]) for i in range(0,len(intervals),2)]
        raw_pair_sets = list(itertools.starmap(kd_query,zip(itertools.repeat(positions),
                                                       interval_shells)))


        psi = list(itertools.starmap(correlation,zip(raw_pair_sets,
                                                     itertools.repeat(galaxyXYZV))))
        psione = [a[0]/10**4 for a in psi]
        psitwo = [a[1]/10**4 for a in psi]
        a = [a[2] for a in psi]

        common.writedict(outfolder+outfile.format(n+settings['offset'])+'_rawdata.json',{'psione':psione,
                                                                                         'psitwo':psitwo,
                                                                                         'a':a,
                                                                                         'xs':xs})

        

#@profile
def correlation(interval_shell,galaxies):
    # galaxies = [(galaxies[a],galaxies[b]) for a,b in interval_shell] 
    galaxyPairs = np.array(list(interval_shell))
    lGalaxies = galaxies[galaxyPairs[:,0]]
    rGalaxies = galaxies[galaxyPairs[:,1]]

    #"Galaxy 1 VelocitieS"
    g1vs = lGalaxies[:,3]
    g2vs = rGalaxies[:,3]
    g1pos = lGalaxies[:,0:3]
    g2pos = rGalaxies[:,0:3]

    g1dist = np.linalg.norm(g1pos,axis=1)
    g2dist = np.linalg.norm(g2pos,axis=1)

    #Normalized galaxy position vectors
    #The extra [:,None] is there to make the denominator axis correct. See
    #http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    g1norm = g1pos/g1dist[:,None] 
    g2norm = g2pos/g2dist[:,None]
    
    distBetweenG1G2 = np.linalg.norm(g2pos-g1pos,axis=1)
    r = (g2pos-g1pos) / distBetweenG1G2[:,None]

    cosdTheta = inner1d(g1norm,g2norm)
    cosTheta1 = inner1d(r,g1norm)
    cosTheta2 = inner1d(r,g2norm)
    
    psiOneNum = psiOneNumerator(g1vs,g2vs,cosdTheta).sum()
    psiOneDen = psiOneDenominator(cosdTheta).sum()
    psiTwoNum = psiTwoNumerator(g1vs,g2vs,cosTheta1,cosTheta2).sum()
    psiTwoDen = psiTwoDenominator(cosTheta1,cosTheta2,cosdTheta).sum()
    aNum = aNumerator(cosdTheta,g1dist,g2dist,distBetweenG1G2).sum()
    aDen = aDenominator(cosdTheta,distBetweenG1G2).sum()
    
    psione = psiOneNum/psiOneDen
    psitwo = psiTwoNum/psiTwoDen
    a = aNum/aDen
    return (psione,psitwo,a)

# @profile
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
    kd = cKDTree(positions)
    #It's great that you can subtract sets! Subtracting set a from set b removes everything in set a from set b
    #so if a=[1,2,3,4,5] and b=[2,4,6,8,10], a-b=[1,3,5]
    one = kd.query_pairs(interval[0])
    two = kd.query_pairs(interval[1])
    one.difference_update(two) # Use this instead of one - two because of time complexity.
    #see: https://wiki.python.org/moin/TimeComplexity
    return one

def stats(args):

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
        pylab.axis((0,31,0,70))
        pylab.legend()

        with pdfback.PdfPages(outfolder+outfile.format(n+settings['offset'])) as pdf:
            pdf.savefig(fig)
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

    


