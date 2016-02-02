#from __future__ import (absolute_import, division,
#                        print_function, unicode_literals)
#STERN WARNING: Running this code on python 2 is not recommended, for many reasons. First, Garbage Collection
#seems greatly improved in python 3, resulting in *much* reduced memory usage. Second, everything runs much faster
#under python 3 for whatever reason.
import common
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import Pool
import itertools
import matplotlib
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
import smtplib




TEMP_DIRECTORY = "/media/christopher/2TB/Christopher/code/Physics/GalaxyMap/tmp/"

def main(args):
    np.seterr(divide='ignore',invalid='ignore')
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
    #Get setup information from the settings file
    settings =   common.getdict(args.settings)
    numpoints =    settings['numpoints']
    dr =           settings['dr']
    min_r =        settings['min_r']
    orig_outfile = settings['output_file_name']
    step_type =    settings['step_type']
    infile =       settings['input_file']
    unitslist =    settings['binunits']
    maxd_master =  settings['max_distance']
    pool = Pool(processes=6)
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

                #Warning: This assumes that the infiles don't ever change. Make sure to clean your cache regularly!
                #HistogramFiles are the places where histograms from the singleHistogram function are stored.
            else:
                inFileList = [rawInFile]
            histogramFiles = [[TEMP_DIRECTORY + 'histogram_binsize-{}_{}.npy'.format(distargs[0],
                                                                                     formatHash(oneInFile))
                               for distargs in distance_args]
                              for oneInFile in inFileList]

            if args.comp:
                print('computing...')
                try:
                    nothing = pool.starmap(compute,zip(inFileList,
                                                       itertools.repeat(maxd),
                                                       itertools.repeat(units)
                                                   )) 
                except AttributeError:
                    nothing = pool.map(starCompute,zip(inFileList, 
                                                       itertools.repeat(maxd),
                                                       itertools.repeat(units)
                                                   ))
                    list(nothing) #stupid lazy functions. I tell ya... Back in my day, functions HAD A WORK ETHIC!
                 #They did WHAT they were told WHEN they were told to do it!
                #This line is only required for using itertools starmap with the compute function
                print(" - Done!")
            if args.hist:
                print('Histogramming...')
                try:
                    
                    hashes = pool.starmap(getHash, zip(inFileList,
                                                       itertools.repeat(units)))
                except AttributeError:
                    hashes = pool.map(starGetHash,zip(inFileList,itertools.repeat(units)))
                    #the lazy idiom works for me here!
                
                nothing = itertools.starmap(histogram,zip(hashes,
                                                          itertools.repeat(xs),
                                                          itertools.repeat(intervals),
                                                          histogramFiles))
                list(nothing) #because itertools starmap is LAZY
                print(" - Done")
            if args.plots:
                print('plotting...')
                params = list(itertools.product(infileindices, [dist_arg[0] for dist_arg in distance_args]))
                hists = [TEMP_DIRECTORY + 'histogram_binsize-{}_{}.npy'.format(param[1],
                                                                               formatHash(rawInFile,param[0]))
                         for param in params]
                
                outfiles = [outfile.format(param[0],param[1],units) for param in params]
                
                try:
                    pool.starmap(stats,zip(outfiles,
                                           hists,
                                           itertools.repeat(units)))
                except AttributeError:
                    pool.map(starStats,zip(outfiles,
                                                     hists,
                                                     itertools.repeat(units)))
            if args.stats:
                print('statting..?')
                print('no, that doesn\'t sound right')
                print('computing statistics...')
                for histogramFilesList,distanceParameters in zip(map(list, zip(*histogramFiles)),distance_args):
                    standBackStats(histogramFilesList,
                                   readName,
                                   units,
                                   outfile.format('',distanceParameters[0],units.replace('/','')),
                                   maxd=maxd_master
                    )
                print('stats saved in {}.pdf.'.format(outfile.format('','<dist>','<units>')) )
def starGetHash(x):
    return getHash(*x)

def starCompute(x):
    return compute(*x)

def starStats(x):
    return stats(*x)
    
def formatHash(string,*args):
    return hashlib.md5(string.format(*args).encode('utf-8')).hexdigest()   

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
        data = correlation(galaxyXYZV,maxd)
    except RuntimeError:
        print('Runtime Error encountered at {}.'.format(infile))
        raise
    return data

#@profile 
def _kd_query(positions,maxd):
    """Returns a np array of pairs of galaxies."""
    #This is still the best function, despite all of my scheming.
    tmpfilename = TEMP_DIRECTORY + 'rawkd_{}_{}.npy'.format(maxd,myNpHash(positions))
    #Warning: There might be more hash collisions because of this string ^ conversion. Hopefully not.
    #THERE WERE. Thanks for just leaving a warning instead of fixing it :P
    #The warning still stands, but it's a bit better now.
    
    if os.path.exists(tmpfilename):
        print("!",end="",flush=True)
        return np.load(tmpfilename)
    else:
        print(".",end="",flush=True)
        #sys.stdout.flush()
        kd = cKDTree(positions)
        pairs = kd.query_pairs(maxd)
        #print(len(pairs))
        #print((kd.count_neighbors(kd,max(intervals))-len(positions))/2)
        #Find zero-distance pairs
        removePairs = kd.query_pairs(np.finfo(float).eps)
        pairs.difference_update(removePairs)
        listOfPairs = list(pairs)
        pairarray = np.array(listOfPairs) #The np array creation takes a LOT of time. I am still wondering why.
        del pairs, removePairs, kd #This also takes forever.
        gc.collect()
        np.save(tmpfilename,pairarray)
        return pairarray
    

# def ss(numpyarray):
#     print(numpyarray.nbytes)

# def gs(numpyarray):
#     return numpyarray.nbytes


def correlation(galaxies,maxd,usewt=False):
    """ Computes the raw galaxy-galaxy correlation information on a per-galaxy basis, and saves it to file"""
    #There are lots of dels in this function because otherwise it tends to gobble up memory.
    #I think there might be a better way to deal with the large amounts of memory usage, but I don't yet
    #know what it is.
    tmpfilename = TEMP_DIRECTORY+'plotData_{}.npy'.format(myNpHash(galaxies))
    if os.path.exists(tmpfilename):
        print("*",end="",flush=True)
        return
                                                          
    # galaxies = [(galaxies[a],galaxies[b]) for a,b in interval_shell]
    galaxyPairs = _kd_query(galaxies[:,0:3],maxd)
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
    #ss(r)
    #print(gs(g1vs)+gs(g2vs)+gs(g1norm)+gs(g2norm)+gs(distBetweenG1G2)+gs(r)+gs(cosdTheta)+gs(cosTheta1)+gs(cosTheta2))
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
    # np.savez_compressed(TEMP_DIRECTORY+'plotData_{}.npz'.format(hashlib.md5(str(galaxies).encode('utf-8')).hexdigest()),
    #                     p1n = indPsiOneNum,
    #                     p1d = indPsiOneDen,
    #                     p2n = indPsiTwoNum,
    #                     p2d = indPsiTwoDen,
    #                     an  = indAFeldmanNum,
    #                     ad  = indAFeldmanDen,
    #                     bn  = indBNum,
    #                     bd  = indBDen,
    #                     dist= distBetweenG1G2
    # )
    np.save(tmpfilename,
            np.array([indPsiOneNum,
                     indPsiOneDen,
                     indPsiTwoNum,
                     indPsiTwoDen,
                     indAFeldmanNum,
                     indAFeldmanDen,
                     indBNum,
                     indBDen,
                     distBetweenG1G2
            ])
    )
    
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

def histogram(theHash,xs,intervals,writeOut):
    """Bins individual galaxy-pair data by distance, and writes the results to a np array file.
    Loading the data is pretty slow, so now we do that only once, then do a histogram on it a bunch of times.

    Previously, we had to load the file and do one histogram, then load the file and do one histogram...
    We'll save 18 * 100 load cycles this way!
    """
    # try:
    print("y",end="",flush=True)
    data =np.load(TEMP_DIRECTORY+'plotData_{}.npy'.format(theHash))
        #sys.stdout.flush()
    # except IOError:
    #     #Then the file is saved in the old, klunky npz format. Let's go ahead and load it, then save it to an npy
    #     #for waaay faster loading next time.
    #     with np.load(TEMP_DIRECTORY+'plotData_{}.npz'.format(theHash)) as tdata: #T is for temporary. 
        
    #         data = np.array([tdata['p1n'],
    #                          tdata['p1d'],
    #                          tdata['p2n'],
    #                          tdata['p2d'],
    #                          tdata['an'],
    #                          tdata['ad'],
    #                          tdata['bn'],
    #                          tdata['bd'],
    #                          tdata['dist']
    #                      ])
    #         print("z",end="",flush=True)
    #         #sys.stdout.flush()
    #     np.save(TEMP_DIRECTORY + 'plotData_{}.npy'.format(theHash),data) #npy files are SO MUCH faster than npz
    #if not manysq:
    #    xs = [xs]
    #    intervals = [intervals]
    #    writeOut = [writeOut]
    #print(xs, intervals, writeOut)
    parameters = list(zip(itertools.repeat(data),xs,intervals,writeOut))
    nothing = itertools.starmap(singleHistogram,parameters)
    list(nothing) #I HATE LAZY FUNCTIONS most of the time

def singleHistogram(data,xs,intervals,writeOut):
    if os.path.exists(writeOut):
        print("*",end="",flush=True)
        return
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
    finalResults = np.array( (psione,psitwo,a,b,psiParallel,psiPerpindicular,xs) )
    np.save(writeOut,finalResults)

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

def standBackStats(a,b,c,d,maxd=100):
    standBackStats_toomanyfiles(a,b,c,d,maxd=maxd)
    #standBackStats_allonepage(a,b,c,d+'.pdf',maxd=maxd)
    #Set here what you want the stats routine to do. Right now I'm prepping a minipaper, so I need lots of
    #single plots that can fit on page instead of lots of plots glued together.
    
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
    arrrghs = common.parseCmdArgs([['settings'],
                                   ['-c','--comp'],
                                   ['-H','--hist'],
                                   ['-p','--plots'],
                                   ['-s','--stats'],
                                   ['-n','--notify']],
                                  ['Settings json file',
                                   'Compute values for individual galaxies',
                                   'Compute histograms (requires a prior or concurrent -c run)',
                                   'Make a plot for every input survey (requires a prior or concurrent -H run)',
                                   'Do the overview stats routine, one plot for all surveys (requires a prior or concurrent -H run)',
                                   'Notify me via text message when the job is done running'
                                  ],
                                   [str,'bool','bool','bool','bool','bool'])
    try:
        main(arrrghs)
        if arrrghs.notify:
            sendMessage("Job Finished")
    except:
        if arrrghs.notify:
            sendMessage("Job Failed")
        raise



