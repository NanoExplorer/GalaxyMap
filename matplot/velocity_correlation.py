#from __future__ import (absolute_import, division,
#                        print_function, unicode_literals)
#STERN WARNING: Running this code on python 2 is not recommended, for many reasons. First, Garbage Collection
#seems greatly improved in python 3, resulting in *much* reduced memory usage. Second, everything runs much faster
#under python 3 for whatever reason.
import matplotlib
import common
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import Pool
import itertools
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfback
import os
import hashlib
#import pdb
from numpy.core.umath_tests import inner1d #Note: this function has no documentation and could easily be deprecated.
#if that happens, you can always use the syntax (a*b).sum(axis=1), which is ever so slightly slower and much more
#ugly.
#Alternate definition:
# def inner1d(a,b):
#     return (a*b).sum(axis=1)
import gc
import time
import smtplib
#import matplotlib.ticker as mtick

"""
Traceback (most recent call last):
  File "velocity_correlation.py", line 963, in <module>
    main(arrrghs)
  File "velocity_correlation.py", line 99, in main
    itertools.repeat(maxd_master))
  File "velocity_correlation.py", line 411, in standBackStats_perfectBackground
    std = np.std(allData,axis=0)
  File "/home/christopher/.Envs/new-matplot/lib/python3.4/site-packages/numpy/core/fromnumeric.py", line 2985, in std
    keepdims=keepdims)
  File "/home/christopher/.Envs/new-matplot/lib/python3.4/site-packages/numpy/core/_methods.py", line 124, in _std
    keepdims=keepdims)
  File "/home/christopher/.Envs/new-matplot/lib/python3.4/site-packages/numpy/core/_methods.py", line 77, in _var
    arr = asanyarray(a)
  File "/home/christopher/.Envs/new-matplot/lib/python3.4/site-packages/numpy/core/numeric.py", line 525, in asanyarray
    return array(a, dtype, copy=False, order=order, subok=True)
ValueError: could not broadcast input array from shape (7,50) into shape (7)
"""


TEMP_DIRECTORY = "/media/christopher/2TB/Christopher/code/Physics/GalaxyMap/tmp/"
PERFECT_LOCATION = "output/PERFECT_DONTTOUCH/COMPOSITE-MOCK-bin-{:.0f}-{}.npy"
print("Warning: Non-general perfect location")
def main(args):
    np.seterr(divide='ignore',invalid='ignore')
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
    #Get setup information from the settings files
    settings =   common.getdict(args.settings)
    numpoints =    settings['numpoints']
    dr =           settings['dr']
    min_r =        settings['min_r']
    orig_outfile = settings['output_file_name']
    step_type =    settings['step_type']
    infile =       settings['input_file']
    unitslist =    settings['binunits']
    maxd_master =  settings['max_distance']
    pool = Pool(processes=2)
    if settings['many_squared']:
        distance_args_master = list(zip(dr,min_r,numpoints))
        file_schemes  = list(zip(infile,orig_outfile,settings['readable_name']))
        xintervals = [common.genBins(x[1],x[2],x[0],step_type) for x in distance_args_master]
        xs_master = [a[0] for a in xintervals]
        intervals_master = [a[1] for a in xintervals]
    else:
        #Everything is built around lists now, so we just build lists of length one!
        distance_args_master = [(dr,min_r,numpoints)]
        file_schemes = [(infile,orig_outfile,settings['readable_name'])]
        xs_master,intervals_master = common.genBins(min_r,numpoints,dr,step_type)
        xs_master = [xs_master]
        intervals_master = [intervals_master]
        
    infileindices = [x + settings['offset'] for x in range(settings['num_files'])]
    for rawInFile, outfile, readName in file_schemes:
        for units in unitslist:
            if units == 'km/s':
                
                xs = [[x * 100 for x in y] for y in xs_master]
                intervals = [[x * 100 for x in y] for y in intervals_master]
                distance_args = [(x[0]*100,x[1]*100,x[2]) for x in distance_args_master]
                maxd = maxd_master * 100
            else:
                xs = xs_master
                intervals = intervals_master
                distance_args = distance_args_master
                maxd = maxd_master
                
            if settings['many']:
                #If there are lots of files, set them up accordingly.
                inFileList = [rawInFile.format(x) for x in infileindices]
                
            else:
                inFileList = [rawInFile]
                
            with Pool(processes=2) as pool:
                histogramData = list(pool.starmap(turboRun,zip(inFileList,
                                                                   itertools.repeat(maxd),
                                                                   itertools.repeat(units),
                                                                   itertools.repeat(xs),
                                                                   itertools.repeat(intervals))))
                """
                Each turbo run returns a list of histograms [ 5-length histogram, 10-length histogram, 20-length etc]
                so histogramData is a list of turbo runs, which means data is a jagged array
                data = [
                
                [ [ ----- ],
                  [ ---------- ],
                  [ -------------------- ] ],
                
                [ [ ----- ],
                  [ ---------- ],
                  [ -------------------- ] ],
                
                ]
                """
            for scheme_index in range(len(intervals)):
                hist_for_scheme = np.array([turbo_data[scheme_index] for turbo_data in histogramData])
                standBackStats_perfectBackground(hist_for_scheme,
                                                 readName,
                                                 units,
                                                 outfile.format('',distance_args[scheme_index][0],units.replace('/','')),
                                                 PERFECT_LOCATION,
                                                 maxd
                                             )
            print(" Done!")
    
def formatHash(string,*args):
    return hashlib.md5(string.format(*args).encode('utf-8')).hexdigest()

def turboRun(infile,maxd,units,xs,intervals):
    """Returns a list of histograms, where a histogram is defined as a numpy array
    [[psi1 values] [psi2 values]
     [a values]    [b values]
     [par values]  [perp values]
     [x values]
    ]

    The list of histograms returned is as follows
    [histogram(xs[0],intervals[0]),
     histogram(xs[1],intervals[1]),
     ...
     histogram(xs[n],intervals[n])
    ]
    """
    correlations=compute(infile,maxd,units)
    return list(itertools.starmap(singleHistogram,zip(itertools.repeat(correlations),
                                                xs,
                                                intervals
                                            )
                                 )
               )
    
def compute(infile,maxd,units):
    """Formerly called 'singlePlot,' this function computes all pair-pair galaxy information,
    including figuring out the list of pairs.
    Input - cf2 file to read from.
          - Intervals (to determine maximum value to find pairs to).
          - units (km/s or Mpc/h) to determine which quantities from input data we should use
    Output - None
    Side effects - writes a file in the /tmp directory containing raw pair-pair information
         This file is called plotData_<HASH>.npz, where the hash is an md5 hash of the str output of the
         'galaxyXYZV' np array.
                 - writes a file in the /tmp directory containing a list of pairs.
         This file is called rawkd_<LENGTH-WE-VISITED-TO>_<HASH>.npy where the hash is determined in the same way
         as above.

    Future improvements - Maybe return the hash instead of nothing, so we don't have to rebuild the hashes later?
    Maybe index all hashes into a helpful file in the tmp directory?
    """
    #Load the survey
    galaxies = common.loadData(infile, dataType = 'CF2')

    #Make an array of just the x,y,z coordinate and radial component of peculiar velocity (v)
    if units == 'Mpc/h':
        galaxyXYZV = np.array([(a.x,a.y,a.z,a.v,a.dv) for a in galaxies])
    elif units == 'km/s':
        galaxyXYZV = np.array([a.getRedshiftXYZ() + (a.v,a.dv) for a in galaxies])
        #You can concatenate tuples. getRedshiftXYZ returnes a tuple, and I just append a.v (and dv) to it.
    else:
        print("I TOLD YOU, ONLY USE km/s or Mpc/h as your units!!!")
        print("And you should definitely, NEVER EVER, use '{}'!!".format(units))

    try:
        data = correlation(galaxyXYZV,maxd,units)
    except RuntimeError:
        print('Runtime Error encountered at {}.'.format(infile))
        raise
    return data

#@profile 
def _kd_query(positions,maxd,units):
    """Returns a np array of pairs of galaxies."""
    #This is still the best function, despite all of my scheming.
    tmpfilename = TEMP_DIRECTORY + 'rawkd_{}_{}.npy'.format(maxd,myNpHash(positions))
    #Warning: There might be more hash collisions because of this string ^ conversion. Hopefully not.
    #THERE WERE. Thanks for just leaving a warning instead of fixing it :P
    #The warning still stands, but it's a bit better now.
    
    if units == "km/s" and os.path.exists(tmpfilename):
        print("!",end="",flush=True)
        return np.load(tmpfilename)
    else:
        print(".",end="",flush=True)
        kd = cKDTree(positions)
        pairs = kd.query_pairs(maxd)
        #Find zero-distance pairs
        removePairs = kd.query_pairs(np.finfo(float).eps)
        pairs.difference_update(removePairs)
        listOfPairs = list(pairs)
        pairarray = np.array(listOfPairs) #The np array creation takes a LOT of time. I am still wondering why.
        del pairs, removePairs, kd #This also takes forever.
        gc.collect()
        if units == "km/s":
            np.save(tmpfilename,pairarray)
            #The caching scheme only helps if we have the same set of distance data over and over again.
            #That is the case with redshift-binned data, but not anything else.
        return pairarray


def correlation(galaxies,maxd,units,usewt=False):
    """ Computes the raw galaxy-galaxy correlation information on a per-galaxy basis, and saves it to file"""
    #There are lots of dels in this function because otherwise it tends to gobble up memory.
    #I think there might be a better way to deal with the large amounts of memory usage, but I don't yet
    #know what it is.
                                                          
    # galaxies = [(galaxies[a],galaxies[b]) for a,b in interval_shell]
    galaxyPairs = _kd_query(galaxies[:,0:3],maxd,units)
    #print("Done! (with the thing)")
    lGalaxies = galaxies[galaxyPairs[:,0]]
    rGalaxies = galaxies[galaxyPairs[:,1]]
    del galaxyPairs
    
    #"Galaxy 1 VelocitieS"
    g1vs  = lGalaxies[:,3]
    g2vs  = rGalaxies[:,3]
    if usewt:
        wt = 1/((lGalaxies[:,4]**2 + 150**2)*(rGalaxies[:,4]**2 + 150**2))
    else:
        wt = 1
    g1pos = lGalaxies[:,0:3]
    g2pos = rGalaxies[:,0:3]
    del lGalaxies, rGalaxies
    
    #Distances from galaxy to center
    g1dist = np.linalg.norm(g1pos,axis=1)
    g2dist = np.linalg.norm(g2pos,axis=1)
    
    #Normalized galaxy position vectors
    #The extra [:,None] is there to make the denominator axis correct. See
    #http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    g1norm = g1pos/g1dist[:,None]
    del g1dist
    g2norm = g2pos/g2dist[:,None]
    del g2dist
    
    distBetweenG1G2 = np.linalg.norm(g2pos-g1pos,axis=1)
    
    #r is a unit vector pointing from g2 to g1
    r = (g2pos-g1pos) / distBetweenG1G2[:,None]
    del g1pos, g2pos
    
    cosTheta1 = inner1d(r,g1norm)
    cosTheta2 = inner1d(r,g2norm)

    del r #R hogs a lot of memory, so we want to get rid of it as fast as possible.
    cosdTheta = inner1d(g1norm,g2norm)
    del g1norm, g2norm
    #The 'ind' stands for individual
    indPsiOneNum   = wt*psiOneNumerator(g1vs,g2vs,cosdTheta)
    indPsiOneDen   = wt*psiOneDenominator(cosdTheta)
    indPsiTwoNum   = wt*psiTwoNumerator(g1vs,g2vs,cosTheta1,cosTheta2)
    del g1vs, g2vs
    indPsiTwoDen   = wt*psiTwoDenominator(cosTheta1,cosTheta2,cosdTheta)
    indAFeldmanNum = wt*aFeldmanNumerator(cosTheta1,cosTheta2,cosdTheta)
    indAFeldmanDen = wt*aFeldmanDenominator(cosdTheta)
    indBNum        = wt*bNumerator(cosTheta1,cosTheta2)
    indBDen        = wt*bDenominator(cosTheta1,cosTheta2,cosdTheta)
    del cosTheta1, cosTheta2, cosdTheta,wt

    return  np.array([indPsiOneNum,
                     indPsiOneDen,
                     indPsiTwoNum,
                     indPsiTwoDen,
                     indAFeldmanNum,
                     indAFeldmanDen,
                     indBNum,
                     indBDen,
                     distBetweenG1G2
            ])
    
    
#NOTE: If the information passed as 'galaxies' to "correlation" changes, you have to update this getHash function too!
def myNpHash(data):
    return hashlib.md5((str(data)+str(len(data))).encode('utf-8')).hexdigest()

def getHash(filename,units):
    """Loads up CF2 files and uses them to rebuild the hash database.
    Returns a list of strings. The strings should be hashed with hashlib.md5(string.encode('utf-8')).hexdigest()
    I'm not sure what I meant when I put that second line there..."""
    galaxies = common.loadData(filename, dataType = 'CF2')
    if units == 'Mpc/h':
        return myNpHash(np.array([(a.x,a.y,a.z,a.v,a.dv) for a in galaxies]))
    elif units == 'km/s':
        return myNpHash(np.array([a.getRedshiftXYZ() + (a.v,a.dv) for a in galaxies]))
    else:
        raise ValueError("Value of 'units' must be 'Mpc/h' or 'km/s'. Other unit schemes do not exist at present")


def singleHistogram(data,xs,intervals):
    indPsiOneNum = data[0]
    indPsiOneDen = data[1]
    indPsiTwoNum = data[2]
    indPsiTwoDen = data[3]
    indANum = data[4]
    indADen = data[5]
    indBNum = data[6]
    indBDen = data[7]
    distBetweenG1G2 = data[8]

    psiOneNum = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiOneNum)[0]
    psiOneDen = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiOneDen)[0] 
    psiTwoNum = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiTwoNum)[0] 
    psiTwoDen = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiTwoDen)[0]
    aNum      = np.histogram(distBetweenG1G2,bins = intervals,weights = indANum)[0]
    aDen      = np.histogram(distBetweenG1G2,bins = intervals,weights = indADen)[0]
    bNum      = np.histogram(distBetweenG1G2,bins = intervals,weights = indBNum)[0]
    bDen      = np.histogram(distBetweenG1G2,bins = intervals,weights = indBDen)[0]
    
    psione = psiOneNum/psiOneDen
    psitwo = psiTwoNum/psiTwoDen
    del psiOneNum, psiOneDen, psiTwoNum, psiTwoDen
    a = aNum/aDen
    b = bNum/bDen
    del aNum, aDen, bNum, bDen
    
    aminusb = (a-b)
    psiParallel = ((1-b)*psione-(1-a)*psitwo)/aminusb
    psiPerpindicular = (a*psitwo-b*psione)/aminusb
    del aminusb
    return np.array( (psione,psitwo,a,b,psiParallel,psiPerpindicular,xs) )

def psiOneNumerator(rv1, rv2, cosdTheta):
    """
    Calculates \psi 1's numerator as defined in the Gorski paper
    This is one iteration. So call this function a billion times then add up all the results.
    Or, call this function once using numPy arrays!
    """
    return rv1*rv2*cosdTheta

def psiOneDenominator(cosdTheta):
    return (cosdTheta)**2

def psiTwoNumerator(rv1,rv2,costheta1,costheta2):
    return rv1*rv2*costheta1*costheta2

def psiTwoDenominator(costheta1,costheta2,cosdTheta):
    return costheta1*costheta2*cosdTheta

def aFeldmanNumerator(costheta1,costheta2,cosdTheta):
    return costheta1*costheta2*cosdTheta

def aFeldmanDenominator(cosdTheta):
    return cosdTheta**2

def bNumerator(costheta1,costheta2):
    return (costheta1**2)*(costheta2**2)
    
def bDenominator(costheta1,costheta2,cosdTheta):
    return costheta1*costheta2*cosdTheta

#@profile
def stats(writeOut,readIn,units):
    """Make plots of the data output by the main function
    """
    Y_SCALE_FACTOR = 10**4
    #Get settings
    data = np.load(readIn)
    xs = data[6]
    a = data[2]
    b = data[3]
    psione = data[0]/Y_SCALE_FACTOR
    psitwo = data[1]/Y_SCALE_FACTOR
    psipar = data[4]/Y_SCALE_FACTOR
    psiprp = data[5]/Y_SCALE_FACTOR

    fig = plt.figure()
    plt.plot(xs,a,'-',label='$\cal A$ (Borgani)')
    plt.plot(xs,b,'k--',label='$\cal B$')
    plt.title('Moment of the selection function')
    plt.ylabel('Value (unitless)')
    plt.xlabel('Distance, {}'.format(units))
    plt.legend(loc=2)
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.axis((0,31,.62,.815))


    fig2 = plt.figure()
    plt.plot(xs,psione,'-',label='$\psi_1$')
    plt.plot(xs,psitwo,'k--',label="$\psi_2$")
    plt.title('Velocity correlation function')
    plt.xlabel('Distance, {}'.format(units))
    plt.ylabel('Correlation, $10^4 (km/s)^2$')
    #plt.axis((0,31,0,32))
    plt.legend()

    fig3 = plt.figure()
    plt.plot(xs,psipar,'-',label='$\psi_{\parallel}$')
    plt.plot(xs,psiprp,'-',label='$\psi_{\perp}$')
    plt.title('Velocity correlation')
    plt.xlabel('Distance, {}'.format(units))
    plt.ylabel('Correlation, $10^4 (km/s)^2$')
    plt.legend()

    with pdfback.PdfPages(writeOut) as pdf:
        pdf.savefig(fig3)
        pdf.savefig(fig2)
        pdf.savefig(fig)
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    np.save(writeOut+".npy",data)

def standBackStats(a,b,c,d,maxd=100):
    standBackStats_toomanyfiles(a,b,c,d,maxd=maxd,savenpy=True)
    #standBackStats_allonepage(a,b,c,d+'.pdf',maxd=maxd)
    #Set here what you want the stats routine to do. Right now I'm prepping a minipaper, so I need lots of
    #single plots that can fit on page instead of lots of plots glued together.

def standBackStats_perfectBackground(allData,name,units,writeOut,perfect_location,maxd,savenpy=True):
    #allData = np.array(list(map(np.load, inFileList)))
    #One inFile contains the following: [p1, p2, a, b, psiparallel, psiperpindicular]
    xs = allData[0][6]
    perfect = np.load(perfect_location.format(xs[1]-xs[0],units.replace('/','')))
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)
    low68 = perfect[range(3,37,6)] # I don't know why I saved them in this order, but at least it's not too hard
    hi68  = perfect[range(4,37,6)] # to extract.
    low95 = perfect[range(5,37,6)]
    hi95  = perfect[range(6,37,6)]


    plotName = name
    """
    matplotlib.rc('font',size=10)
    
    f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharex='col',sharey='row',figsize=(8.5,11))
    f.suptitle('Statistics of the {} Survey Mocks'.format(plotName))
    ax1.errorbar(xs,
                 avg[0]/10**4,
                 yerr=std[0]/10**4,
                 fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5
    )
    ax1.fill_between(xs,low68[0]/10**4,hi68[0]/10**4,facecolor='black',alpha=0.25)
    ax1.fill_between(xs,low95[0]/10**4,hi95[0]/10**4,facecolor='black',alpha=0.25)
    ax1.set_title('$\psi_1$ Correlation')
    #ax1.set_xlabel('Distance, Mpc/h')
    ax1.set_ylabel('Correlation, $10^4 (km/s)^2$')
    #ax1.axis(correlationScale)
    
    ax2.errorbar(xs, avg[1]/10**4, yerr=std[1]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax2.set_title('$\psi_2$ Correlation')
    ax2.fill_between(xs,low68[1]/10**4,hi68[1]/10**4,facecolor='black',alpha=0.25)
    ax2.fill_between(xs,low95[1]/10**4,hi95[1]/10**4,facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    #plt.ylabel('Correlation, $(km/s)^2$')
    #plt.axis(correlationScale)
    
    ax3.errorbar(xs, avg[2], yerr=std[2], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax3.set_title('Moment of the Selection Function, $\cal A$')
    ax3.fill_between(xs,low68[2],hi68[2],facecolor='black',alpha=0.25)
    ax3.fill_between(xs,low95[2],hi95[2],facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    ax3.set_ylabel('Value (unitless)')
    #plt.axis(momentScale)
    
    ax4.errorbar(xs, avg[3], yerr=std[3], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax4.set_title('Moment of the Selection Function, $\cal B$')
    ax4.fill_between(xs,low68[3],hi68[3],facecolor='black',alpha=0.25)
    ax4.fill_between(xs,low95[3],hi95[3],facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    #plt.ylabel('Value (unitless)')
    #plt.axis(momentScale)

    ax5.errorbar(xs, avg[4]/10**4, yerr=std[4]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax5.fill_between(xs,low68[4]/10**4,hi68[4]/10**4,facecolor='black',alpha=0.25)
    ax5.fill_between(xs,low95[4]/10**4,hi95[4]/10**4,facecolor='black',alpha=0.25)
    ax5.set_title('$\Psi_{{\parallel}}$ Correlation')
    ax5.set_xlabel('Distance, {}'.format(units))
    ax5.set_ylabel('Correlation, $10^4 (km/s)^2$')
    #plt.axis(correlationScale)

    ax6.errorbar(xs,avg[5]/10**4, yerr=std[5]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax6.fill_between(xs,low68[5]/10**4,hi68[5]/10**4,facecolor='black',alpha=0.25)
    ax6.fill_between(xs,low95[5]/10**4,hi95[5]/10**4,facecolor='black',alpha=0.25)
    ax6.set_title('$\Psi_{{\perp}}$ Correlation')
    ax6.set_xlabel('Distance, {}'.format(units))
    #plt.ylabel('Correlation, $(km/s)^2$')
    #ax6.axis(correlationScale)

    if units == 'km/s':
        ax5.set_xbound(0,maxd*100)
        ax6.set_xbound(0,maxd*100)
    else:
        ax5.set_xbound(0,maxd)
        ax6.set_xbound(0,maxd)
    
    with pdfback.PdfPages(writeOut) as pdf:
        pdf.savefig(f)
    plt.close(f)
    """
    np.save(writeOut,np.array([xs,avg[0],std[0],low68[0],hi68[0],low95[0],hi95[0],
                               avg[1],std[1],low68[1],hi68[1],low95[1],hi95[1],
                               avg[2],std[2],low68[2],hi68[2],low95[2],hi95[2],
                               avg[3],std[3],low68[3],hi68[3],low95[3],hi95[3],
                               avg[4],std[4],low68[4],hi68[4],low95[4],hi95[4],
                               avg[5],std[5],low68[5],hi68[5],low95[5],hi95[5]]))

        

        
def standBackStats_perfectBackground_old(inFileList,name,units,writeOut,perfect_location,savenpy=False,maxd=100):
    
    theMap = map(np.load, inFileList)
    theList = list(theMap)
    allData = np.array(theList)
    #allData = np.array(list(map(np.load, inFileList)))
    #One inFile contains the following: [p1, p2, a, b, psiparallel, psiperpindicular]
    xs = allData[0][6]
    perfect = np.load(perfect_location.format(xs[1]-xs[0],units.replace('/','')))
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)
    low68 = perfect[range(3,37,6)] # I don't know why I saved them in this order, but at least it's not too hard
    hi68  = perfect[range(4,37,6)] # to extract.
    low95 = perfect[range(5,37,6)]
    hi95  = perfect[range(6,37,6)]

     

    # if units == 'km/s':
    #     correlationScale = (0,5000,-1000,5000)
    # else:
    #     correlationScale = (0,50,-1000,5000)

    #momentScale = (0,30,0.25,1.1)
    plotName = name

    matplotlib.rc('font',size=10)
    
    fig1 = plt.figure()
    
    plt.title('$\psi_1$, {} Survey Mock'.format(plotName))
    plt.errorbar(xs,
                 avg[0],
                 yerr=std[0],
                 fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5
    )
    #print(xs)
    #print(low68[0])
    
    plt.fill_between(xs,low68[0][0:50],hi68[0][0:50],facecolor='black',alpha=0.25)
    plt.fill_between(xs,low95[0][0:50],hi95[0][0:50],facecolor='black',alpha=0.25)
    
    plt.xlabel('Distance, {}'.format(units))
    plt.ylabel('Correlation, (km/s)^2$')
    #plt.axis(correlationScale)
    if units == 'km/s':
        plt.xlim(0,5000)
    else:
        plt.xlim(0,50)
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    fig2 = plt.figure()
    plt.errorbar(xs, avg[1], yerr=std[1], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    plt.title('$\psi_2$, {} Survey Mock'.format(plotName))
    plt.fill_between(xs,low68[1][0:50],hi68[1],facecolor='black',alpha=0.25)
    plt.fill_between(xs,low95[1][0:50],hi95[1],facecolor='black',alpha=0.25)
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Correlation, $(km/s)^2$')
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    #plt.axis(correlationScale)
    if units == 'km/s':
        plt.xlim(0,5000)
    else:
        plt.xlim(0,50)

    with pdfback.PdfPages(writeOut+'-1.pdf') as pdf:
        pdf.savefig(fig1)

    with pdfback.PdfPages(writeOut+'-2.pdf') as pdf:
        pdf.savefig(fig2)
        
    plt.close(fig1)
    plt.close(fig2)
    if savenpy:
        np.save(writeOut,np.array([xs,avg[0],std[0],low68[0],hi68[0],low95[0],hi95[0],
                                   avg[1],std[1],low68[1],hi68[1],low95[1],hi95[1],
                                   avg[2],std[2],low68[2],hi68[2],low95[2],hi95[2],
                                   avg[3],std[3],low68[3],hi68[3],low95[3],hi95[3],
                                   avg[4],std[4],low68[4],hi68[4],low95[4],hi95[4],
                                   avg[5],std[5],low68[5],hi68[5],low95[5],hi95[5]]))
    
def standBackStats_toomanyfiles(inFileList,name,units,writeOut,maxd=100,savenpy=False):
    """Do statistics over many input files, for example the three groups of 100 surveys. Average them, plot w/errorbars."""
    assert(len(inFileList) == 100) #Not true in all cases, but sufficient for debugging. REMOVE this line if problems
    theMap = map(np.load, inFileList)
    theList = list(theMap)
    allData = np.array(theList)
    #allData = np.array(list(map(np.load, inFileList)))
    #One inFile contains the following: [p1, p2, a, b, psiparallel, psiperpindicular]
    xs = allData[0][6]
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)
    low68 = np.percentile(allData,16,axis=0)
    hi68  = np.percentile(allData,100-16,axis=0)
    low95 = np.percentile(allData,2.5,axis=0)
    hi95  = np.percentile(allData,100-2.5,axis=0)
    

    # if units == 'km/s':
    #     correlationScale = (0,5000,-1000,5000)
    # else:
    #     correlationScale = (0,50,-1000,5000)

    #momentScale = (0,30,0.25,1.1)
    plotName = name

    matplotlib.rc('font',size=10)
    
    fig1 = plt.figure()
    
    plt.title('$\psi_1$, {} Survey Mock'.format(plotName))
    plt.errorbar(xs,
                 avg[0]/10**4,
                 yerr=std[0]/10**4,
                 fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5
    )
    plt.fill_between(xs,low68[0]/10**4,hi68[0]/10**4,facecolor='black',alpha=0.25)
    plt.fill_between(xs,low95[0]/10**4,hi95[0]/10**4,facecolor='black',alpha=0.25)
    
    plt.xlabel('Distance, {}'.format(units))
    plt.ylabel('Correlation, $10^4 (km/s)^2$')
    #plt.axis(correlationScale)
    if units == 'km/s':
        plt.xlim(0,5000)
    else:
        plt.xlim(0,50)

    fig2 = plt.figure()
    plt.errorbar(xs, avg[1]/10**4, yerr=std[1]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    plt.title('$\psi_2$, {} Survey Mock'.format(plotName))
    plt.fill_between(xs,low68[1]/10**4,hi68[1]/10**4,facecolor='black',alpha=0.25)
    plt.fill_between(xs,low95[1]/10**4,hi95[1]/10**4,facecolor='black',alpha=0.25)
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Correlation, $(km/s)^2$')
    #plt.axis(correlationScale)
    if units == 'km/s':
        plt.xlim(0,5000)
    else:
        plt.xlim(0,50)

    
    with pdfback.PdfPages(writeOut+'-1.pdf') as pdf:
        pdf.savefig(fig1)

    with pdfback.PdfPages(writeOut+'-2.pdf') as pdf:
        pdf.savefig(fig2)
        
    plt.close(fig1)
    plt.close(fig2)
    if savenpy:
        np.save(writeOut,np.array([xs,avg[0],std[0],low68[0],hi68[0],low95[0],hi95[0],
                                   avg[1],std[1],low68[1],hi68[1],low95[1],hi95[1],
                                   avg[2],std[2],low68[2],hi68[2],low95[2],hi95[2],
                                   avg[3],std[3],low68[3],hi68[3],low95[3],hi95[3],
                                   avg[4],std[4],low68[4],hi68[4],low95[4],hi95[4],
                                   avg[5],std[5],low68[5],hi68[5],low95[5],hi95[5]]))
    
def standBackStats_allonepage(inFileList,name,units,writeOut,maxd=100):
    """Do statistics over many input files, for example the three groups of 100 surveys. Average them, plot w/errorbars."""
    assert(len(inFileList) == 100) #Not true in all cases, but sufficient for debugging. REMOVE this line if problems
    theMap = map(np.load, inFileList)
    theList = list(theMap)
    allData = np.array(theList)
    #allData = np.array(list(map(np.load, inFileList)))
    #One inFile contains the following: [p1, p2, a, b, psiparallel, psiperpindicular]
    xs = allData[0][6]
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)
    low68 = np.percentile(allData,16,axis=0)
    hi68  = np.percentile(allData,100-16,axis=0)
    low95 = np.percentile(allData,2.5,axis=0)
    hi95  = np.percentile(allData,100-2.5,axis=0)
    
    #correlationScale = (0,30,0,160000)
    #momentScale = (0,30,0.25,1.1)
    plotName = name

    matplotlib.rc('font',size=10)
    
    f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharex='col',sharey='row',figsize=(8.5,11))
    f.suptitle('Statistics of the {} Survey Mocks'.format(plotName))
    ax1.errorbar(xs,
                 avg[0]/10**4,
                 yerr=std[0]/10**4,
                 fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5
    )
    ax1.fill_between(xs,low68[0]/10**4,hi68[0]/10**4,facecolor='black',alpha=0.25)
    ax1.fill_between(xs,low95[0]/10**4,hi95[0]/10**4,facecolor='black',alpha=0.25)
    ax1.set_title('$\psi_1$ Correlation')
    #ax1.set_xlabel('Distance, Mpc/h')
    ax1.set_ylabel('Correlation, $10^4 (km/s)^2$')
    #ax1.axis(correlationScale)
    
    ax2.errorbar(xs, avg[1]/10**4, yerr=std[1]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax2.set_title('$\psi_2$ Correlation')
    ax2.fill_between(xs,low68[1]/10**4,hi68[1]/10**4,facecolor='black',alpha=0.25)
    ax2.fill_between(xs,low95[1]/10**4,hi95[1]/10**4,facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    #plt.ylabel('Correlation, $(km/s)^2$')
    #plt.axis(correlationScale)
    
    ax3.errorbar(xs, avg[2], yerr=std[2], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax3.set_title('Moment of the Selection Function, $\cal A$')
    ax3.fill_between(xs,low68[2],hi68[2],facecolor='black',alpha=0.25)
    ax3.fill_between(xs,low95[2],hi95[2],facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    ax3.set_ylabel('Value (unitless)')
    #plt.axis(momentScale)
    
    ax4.errorbar(xs, avg[3], yerr=std[3], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax4.set_title('Moment of the Selection Function, $\cal B$')
    ax4.fill_between(xs,low68[3],hi68[3],facecolor='black',alpha=0.25)
    ax4.fill_between(xs,low95[3],hi95[3],facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    #plt.ylabel('Value (unitless)')
    #plt.axis(momentScale)

    ax5.errorbar(xs, avg[4]/10**4, yerr=std[4]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax5.fill_between(xs,low68[4]/10**4,hi68[4]/10**4,facecolor='black',alpha=0.25)
    ax5.fill_between(xs,low95[4]/10**4,hi95[4]/10**4,facecolor='black',alpha=0.25)
    ax5.set_title('$\Psi_{{\parallel}}$ Correlation')
    ax5.set_xlabel('Distance, {}'.format(units))
    ax5.set_ylabel('Correlation, $10^4 (km/s)^2$')
    #plt.axis(correlationScale)

    ax6.errorbar(xs,avg[5]/10**4, yerr=std[5]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax6.fill_between(xs,low68[5]/10**4,hi68[5]/10**4,facecolor='black',alpha=0.25)
    ax6.fill_between(xs,low95[5]/10**4,hi95[5]/10**4,facecolor='black',alpha=0.25)
    ax6.set_title('$\Psi_{{\perp}}$ Correlation')
    ax6.set_xlabel('Distance, {}'.format(units))
    #plt.ylabel('Correlation, $(km/s)^2$')
    #ax6.axis(correlationScale)

    if units == 'km/s':
        ax5.set_xbound(0,maxd*100)
        ax6.set_xbound(0,maxd*100)
    else:
        ax5.set_xbound(0,maxd)
        ax6.set_xbound(0,maxd)
    
    with pdfback.PdfPages(writeOut) as pdf:
        pdf.savefig(f)
    plt.close(f)
    np.save(writeOut,np.array([xs,avg[0],std[0],low68[0],hi68[0],low95[0],hi95[0],
                               avg[1],std[1],low68[1],hi68[1],low95[1],hi95[1],
                               avg[2],std[2],low68[2],hi68[2],low95[2],hi95[2],
                               avg[3],std[3],low68[3],hi68[3],low95[3],hi95[3],
                               avg[4],std[4],low68[4],hi68[4],low95[4],hi95[4],
                               avg[5],std[5],low68[5],hi68[5],low95[5],hi95[5]]))



def standBackStats_yuyu(inFileList,name,units,writeOut,min_r,numpoints,dr):
    """Do statistics over many input files, for example the three groups of 100 surveys. Average them, plot w/errorbars."""
    assert(len(inFileList) == 100) #Not true in all cases, but sufficient for debugging.
    theMap = map(np.load, inFileList)
    theList = list(theMap)
    theList2 = []
    
    
    #One inFile contains the following: [p1, p2, a, b]
    xs,intervals = common.genBins(min_r,numpoints,dr,'lin')
    
    for array in theList:
        p1 = array[:,0]
        p2 = array[:,1]
        a = array[:,2]
        b = array[:,3]
        aminusb = (a-b)
        ppar = ((1-b)*p1-(1-a)*p2)/aminusb
        pprp = (a*p2-b*p1)/aminusb
    
        thing = np.concatenate((np.atleast_2d(p1).T,
                                np.atleast_2d(p2).T,
                                np.atleast_2d(a).T,
                                np.atleast_2d(b).T,
                                np.atleast_2d(ppar).T,
                                np.atleast_2d(pprp).T),axis=1).T
 
        theList2.append(thing)

    allData = np.array(theList2)
    print(allData[0])
 
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)
    low68 = np.percentile(allData,16,axis=0)
    hi68  = np.percentile(allData,100-16,axis=0)
    low95 = np.percentile(allData,2.5,axis=0)
    hi95  = np.percentile(allData,100-2.5,axis=0)
    print(len(xs), avg[0].shape)   
    #correlationScale = (0,30,0,160000)
    #momentScale = (0,30,0.25,1.1)
    plotName = name

    matplotlib.rc('font',size=10)
    
    f, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,sharex='col',sharey='row',figsize=(8.5,11))
    f.suptitle('Statistics of the {} Survey Mocks'.format(plotName))
    ax1.errorbar(xs,
                 avg[0]/10**4,
                 yerr=std[0]/10**4,
                 fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5
    )
    ax1.fill_between(xs,low68[0]/10**4,hi68[0]/10**4,facecolor='black',alpha=0.25)
    ax1.fill_between(xs,low95[0]/10**4,hi95[0]/10**4,facecolor='black',alpha=0.25)
    ax1.set_title('$\psi_1$ Correlation')
    #ax1.set_xlabel('Distance, Mpc/h')
    ax1.set_ylabel('Correlation, $10^4 (km/s)^2$')
    #ax1.axis(correlationScale)
    
    ax2.errorbar(xs, avg[1]/10**4, yerr=std[1]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax2.set_title('$\psi_2$ Correlation')
    ax2.fill_between(xs,low68[1]/10**4,hi68[1]/10**4,facecolor='black',alpha=0.25)
    ax2.fill_between(xs,low95[1]/10**4,hi95[1]/10**4,facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    #plt.ylabel('Correlation, $(km/s)^2$')
    #plt.axis(correlationScale)
    
    ax3.errorbar(xs, avg[2], yerr=std[2], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax3.set_title('Moment of the Selection Function, $\cal A$')
    ax3.fill_between(xs,low68[2],hi68[2],facecolor='black',alpha=0.25)
    ax3.fill_between(xs,low95[2],hi95[2],facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    ax3.set_ylabel('Value (unitless)')
    #plt.axis(momentScale)
    
    ax4.errorbar(xs, avg[3], yerr=std[3], fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax4.set_title('Moment of the Selection Function, $\cal B$')
    ax4.fill_between(xs,low68[3],hi68[3],facecolor='black',alpha=0.25)
    ax4.fill_between(xs,low95[3],hi95[3],facecolor='black',alpha=0.25)
    #plt.xlabel('Distance, Mpc/h')
    #plt.ylabel('Value (unitless)')
    #plt.axis(momentScale)

    ax5.errorbar(xs, avg[4]/10**4, yerr=std[4]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax5.fill_between(xs,low68[4]/10**4,hi68[4]/10**4,facecolor='black',alpha=0.25)
    ax5.fill_between(xs,low95[4]/10**4,hi95[4]/10**4,facecolor='black',alpha=0.25)
    ax5.set_title('$\Psi_{{\parallel}}$ Correlation')
    ax5.set_xlabel('Distance, {}'.format(units))
    ax5.set_ylabel('Correlation, $10^4 (km/s)^2$')
    #plt.axis(correlationScale)

    ax6.errorbar(xs,avg[5]/10**4, yerr=std[5]/10**4, fmt = 'k-',
                 elinewidth=0.5,
                 capsize=2,
                 capthick=0.5)
    ax6.fill_between(xs,low68[5]/10**4,hi68[5]/10**4,facecolor='black',alpha=0.25)
    ax6.fill_between(xs,low95[5]/10**4,hi95[5]/10**4,facecolor='black',alpha=0.25)
    ax6.set_title('$\Psi_{{\perp}}$ Correlation')
    ax6.set_xlabel('Distance, {}'.format(units))
    #plt.ylabel('Correlation, $(km/s)^2$')
    #ax6.axis(correlationScale)

    if units == 'km/s':
        ax5.set_xbound(0,100*100)
        ax6.set_xbound(0,100*100)
    else:
        ax5.set_xbound(0,100)
        ax6.set_xbound(0,100)
    
    with pdfback.PdfPages(writeOut) as pdf:
        pdf.savefig(f)
    plt.close(f)
    np.save(writeOut,np.array([xs,avg[0],std[0],low68[0],hi68[0],low95[0],hi95[0],
                               avg[1],std[1],low68[1],hi68[1],low95[1],hi95[1],
                               avg[2],std[2],low68[2],hi68[2],low95[2],hi95[2],
                               avg[3],std[3],low68[3],hi68[3],low95[3],hi95[3],
                               avg[4],std[4],low68[4],hi68[4],low95[4],hi95[4],
                               avg[5],std[5],low68[5],hi68[5],low95[5],hi95[5]]))

def sendMessage(message):
    server = smtplib.SMTP( "smtp.gmail.com", 587 )
    login = common.getdict('mail.json')
    server.starttls()
    server.login(*(login[0]))
    print(server.sendmail("",login[1],message))
    server.quit()


                      
    
if __name__ == "__main__":
    start = time.time()
    arrrghs = common.parseCmdArgs([['settings'],
                                   ['-c','--comp'],
                                   ['-H','--hist'],
                                   ['-p','--plots'],
                                   ['-s','--stats'],
                                   ['-n','--notify'],
                                   ['-b','--statspb']
                               ],
                                  ['Settings json file',
                                   'Compute values for individual galaxies',
                                   'Compute histograms (requires a prior or concurrent -c run)',
                                   'Make a plot for every input survey (requires a prior or concurrent -H run)',
                                   'Do the overview stats routine, one plot for all surveys (requires a prior or concurrent -H run)',
                                   'Notify me via text message when the job is done running',
                                   'Do the stats with perfect contours in the background'
                                  ],
                                   [str,'bool','bool','bool','bool','bool','bool'])
    try:
        main(arrrghs)
        print("That took {:.1f} s".format(time.time()-start))
        if arrrghs.notify:
            sendMessage("Job Finished in {:,1f} s".format(time.time()-start))
    except:
        if arrrghs.notify:
            sendMessage("Job Failed")
        raise
        



