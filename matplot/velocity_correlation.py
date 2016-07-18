#from __future__ import (absolute_import, division,
#                        print_function, unicode_literals)
#STERN WARNING: Running this code on python 2 is not recommended, for many reasons. First, Garbage Collection
#seems greatly improved in python 3, resulting in *much* reduced memory usage. Second, everything runs much faster
#under python 3 for whatever reason.
#import matplotlib
import common
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import Pool
import itertools
#import matplotlib.pyplot as plt
#import matplotlib.backends.backend_pdf as pdfback
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
import os
#import smtplib
#import matplotlib.ticker as mtick

TEMP_DIRECTORY = "/data/c156r133/tmp/"
NUM_PROCESSES=12
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



#PERFECT_LOCATION = "output/PERFECT_DONTTOUCH/COMPOSITE-MOCK-bin-{:.0f}-{}.npy"
USE_TMP = True
#This saves KD tree data. Turn it to True when doing redhsift surveys or when doing the same survey multiple times.

print("Warning: Non-general perfect location")
def main(args):
    np.seterr(divide='ignore',invalid='ignore')
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
    #Get setup information from the settings files
    settings =   common.getdict(args.settings)
    if settings['num_files'] != 10000 and settings['use_npy']:
        print("Sorry! We can only handle 100x100 surveys. For other uses, use the turbospeed branch instead")
        exit()
    numpoints =    settings['numpoints']
    dr =           settings['dr']
    min_r =        settings['min_r']
    orig_outfile = settings['output_file_name']
    step_type =    settings['step_type']
    infile =       settings['input_file']
    unitslist =    settings['binunits']
    maxd_master =  settings['max_distance']
    numpy =        settings['use_npy']
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
    if numpy:
        print(args.override)
        indices = args.override.split(':')
        a = int(indices[0])
        b = int(indices[1])
        file_schemes = file_schemes[a:b]
        print(file_schemes)
    else:    
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
                
            if settings['many'] and not numpy:
                #If there are lots of files, set them up accordingly.
                inFileList = [rawInFile.format(x) for x in infileindices]
                galaxiess = [common.loadData(f) for f in inFileList]                
                d = [np.array([(g.normx,
                                g.normy,
                                g.normz,
                                g.redx,
                                g.redy,
                                g.redz,
                                g.dv,
                                g.d,
                                g.v) for g in galaxies])]) for galaxies in galaxiess]
                i = 
            elif not numpy:
                galaxies = common.loadData(rawInFile,dataType='CF2')
                d = [np.array([(g.normx,
                                g.normy,
                                g.normz,
                                g.redx,
                                g.redy,
                                g.redz,
                                g.dv,
                                g.d,
                                g.v) for g in galaxies])]
                i = ['nothing']
            else:
                data = np.load(rawInFile)
                #d = [ data[x/100][np.invert(np.isnan(data[x/100][:,0]))] for x in range(10000)] BAD. REALLY BAD.
                nansremoved = [ data[x][np.invert(np.isnan(data[x][:,0]))] for x in range(100)]
                del data
                d = [nansremoved[x//100] for x in range(10000)]
                i = [ x%100  for x in range(10000) ]
                print(d[542].shape)


            with Pool(processes=NUM_PROCESSES) as pool:
                histogramData = list(pool.starmap(turboRun,zip(d,i,np,
                                                               itertools.repeat(maxd),
                                                               itertools.repeat(units),
                                                               itertools.repeat(xs),
                                                               itertools.repeat(intervals)
                                                           )))
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
                saveOutput(hist_for_scheme,outfile.format('',distance_args[scheme_index][0],units.replace('/','')))
            print(" Done!")

def formatHash(string,*args):
    return hashlib.md5(string.format(*args).encode('utf-8')).hexdigest()

def turboRun(data,index,np,maxd,units,xs,intervals):
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
    if np:
        d = np.concatenate((data[:,0:7],data[:,7+index:207:100]),axis=1)
        #mask = np.invert(np.isnan(d[:,0]))
    else:
        d = data
    correlations=compute(d,maxd,units)
    del d
    return list(itertools.starmap(singleHistogram,zip(itertools.repeat(correlations),
                                                xs,
                                                intervals
                                            )
                                 )
               )
    
def compute(data,maxd,units):
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

    #Make an array of just the x,y,z coordinate and radial component of peculiar velocity (v)
    if units == 'Mpc/h':
        galaxyXYZV = np.concatenate((data[:,0:3]*data[:,7][:,None],data[:,8][:,None],data[:,6][:,None]),axis=1)
        #0-3 is direction vector. 7 is distance magnitude.
    elif units == 'km/s':
        galaxyXYZV = np.concatenate((data[:,3:6],data[:,8][:,None],data[:,6][:,None]),axis=1)
        #You can concatenate tuples. getRedshiftXYZ returnes a tuple, and I just append a.v (and dv) to it.
    else:
        print("I TOLD YOU, ONLY USE km/s or Mpc/h as your units!!!")
        print("And you should definitely, NEVER EVER, use '{}'!!".format(units))

    data = correlation(galaxyXYZV,maxd,units)

    return data

#@profile 
def _kd_query(positions,maxd,units):
    
    """Returns a np array of pairs of galaxies."""
    #This is still the best function, despite all of my scheming.
    tmpfilename = TEMP_DIRECTORY + 'rawkd_{}_{}.npy'.format(maxd,myNpHash(positions))
    #Warning: There might be more hash collisions because of this string ^ conversion. Hopefully not.
    #THERE WERE. Thanks for just leaving a warning instead of fixing it :P
    #The warning still stands, but it's a bit better now.
    
    if units == "km/s" and os.path.exists(tmpfilename) and USE_TMP:
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
        if units == "km/s" and USE_TMP:
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
    del galaxies
    
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



def saveOutput(allData,writeOut):
    #allData = np.array(list(map(np.load, inFileList)))
    #One inFile contains the following: [p1, p2, a, b, psiparallel, psiperpindicular]
    xs = allData[0][6]
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)

    np.save(writeOut+'nice',np.array([xs,avg[0],std[0],
                                      avg[1],std[1],
                                      avg[2],std[2],
                                      avg[3],std[3],
                                      avg[4],std[4],
                                      avg[5],std[5]]))
    np.save(writeOut+'all',allData)
        



                      
    
if __name__ == "__main__":
    start = time.time()
    arrrghs = common.parseCmdArgs([['settings'],['-o','--override']
                               ],
                                  ['Settings json file','array indices in the format a:b to extract from infile list'
                                  ],
                                   [str,str])

    main(arrrghs)
    print("That took {:.1f} s".format(time.time()-start))
    #ADD a setting to settings to control whether we're dealing with npy
    #or dat
        



