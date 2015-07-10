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

TEMP_DIRECTORY = "/media/christopher/2TB/Christopher/code/Physics/GalaxyMap/tmp"


def main(args):
    np.seterr(divide='ignore',invalid='ignore')
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
    #Get setup information from the settings file
    settings = common.getdict(args.settings)
    numpoints = settings['numpoints']
    dr =        settings['dr']
    min_r =     settings['min_r']
    outfile   = settings['output_file_name']
    step_type = settings['step_type']
    rawInFile = settings['input_file']
    units =     settings['binunits']

    pool = Pool()              
    if settings['many']:
        #If there are lots of files, set them up accordingly.
        inFileList = [rawInFile.format(x+settings['offset']) for x in range(settings['num_files'])]
        histogramFiles = [TEMP_DIRECTORY + 'histogram_binsize-{}_{}.npy'.format(dr,
                                                                   hashlib.md5(infile.encode('utf-8')).hexdigest()
                                                                  ) for infile in inFileList]
        #Warning: This assumes that the infiles don't ever change. Make sure to clean your cache regularly!
        #HistogramFiles are the places where histograms from the singleHistogram function are stored.
        outfiles = [outfile.format(x+settings['offset']) + '.pdf' for x in range(settings['num_files'])]
    else:
        inFileList = [rawInFile]

    xs,intervals = common.genBins(min_r,numpoints,dr,step_type)
    
    if args.comp:
        print('computing...')
        nothing = pool.starmap(compute,zip(inFileList, #See comment lines below for error fixes
                                 itertools.repeat(intervals),
                                 itertools.repeat(units)
        )) #see comment lines below for error fixes
        #NOTE: If this line gives you trouble (e.g. 'pool doesn't have a member called starmap'), just replace
        #that line with this one:
        #pool.map(computeStar,zip(inFileList,
        list(nothing) #stupid lazy functions. I tell ya...

    if args.hist:
        print('Histogramming...')
        strings = pool.starmap(strload, zip(inFileList,
                                            itertools.repeat(units)))

        #There's a HUGE performance gain for not multithreading this part. Go figure
        nothing = itertools.starmap(singleHistogram,zip(strings,
                                                        itertools.repeat(xs),
                                                        itertools.repeat(intervals),
                                                        histogramFiles
        ))
        list(nothing) #because itertools starmap is LAZY
    if args.plots:
        print('plotting...')
        pool.starmap(stats,zip(outfiles,
                               histogramFiles,
                               itertools.repeat(units)))
    if args.stats:
        print('statting..?')
        print('no, that doesn\'t sound right')
        print('computing statistics...')
        standBackStats(histogramFiles,settings['readable_name'],units,outfile.format('') + '.pdf')


def computeStar(args):
    """Helper function for older systems that do not have the pool.starmap() function available"""
    return compute(*args)
    

def compute(infile,intervals,units):
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
        galaxyXYZV = np.array([(a.x,a.y,a.z,a.v) for a in galaxies])
    elif units == 'km/s':
        galaxyXYZV = np.array([a.getRedshiftXYZ() + (a.v,) for a in galaxies])
        #You can concatenate tuples. getRedshiftXYZ returnes a tuple, and I just append a.v to it.
    #Put just the galaxy positions into one array

    try:
        data = correlation(galaxyXYZV,intervals)
    except RuntimeError:
        print('Runtime Error encountered at {}.'.format(infile))
        raise
    return data

#@profile 
def _kd_query(positions,intervals):
    """Returns a np array of pairs of galaxies."""
    #This is still the best function, despite all of my scheming.
    tmpfilename = TEMP_DIRECTORY + 'rawkd_{}_{}.npy'.format(max(intervals),
                                               hashlib.md5(str(positions).encode('utf-8')).hexdigest())
    #Warning: There might be more hash collisions because of this string ^ conversion. Hopefully not.
    if os.path.exists(tmpfilename):
        print("!",end="",flush=True)
        return np.load(tmpfilename)
    else:
        print(".",end="",flush=True)
        kd = cKDTree(positions)
        pairs = kd.query_pairs(max(intervals))
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


def correlation(galaxies,intervals):
    """ Computes the raw galaxy-galaxy correlation information on a per-galaxy basis, and saves it to file"""
    #There are lots of dels in this function because otherwise it tends to gobble up memory.
    #I think there might be a better way to deal with the large amounts of memory usage, but I don't yet
    #know what it is.
    
    # galaxies = [(galaxies[a],galaxies[b]) for a,b in interval_shell]
    galaxyPairs = _kd_query(galaxies[:,0:3],intervals)
    #print("Done! (with the thing)")
    lGalaxies = galaxies[galaxyPairs[:,0]]
    rGalaxies = galaxies[galaxyPairs[:,1]]
    del galaxyPairs
    
    #"Galaxy 1 VelocitieS"
    g1vs  = lGalaxies[:,3]
    g2vs  = rGalaxies[:,3]
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
    indPsiOneNum   = psiOneNumerator(g1vs,g2vs,cosdTheta)
    indPsiOneDen   = psiOneDenominator(cosdTheta)
    indPsiTwoNum   = psiTwoNumerator(g1vs,g2vs,cosTheta1,cosTheta2)
    del g1vs, g2vs
    indPsiTwoDen   = psiTwoDenominator(cosTheta1,cosTheta2,cosdTheta)
    indAFeldmanNum = aFeldmanNumerator(cosTheta1,cosTheta2,cosdTheta)
    indAFeldmanDen = aFeldmanDenominator(cosdTheta)
    indBNum        = bNumerator(cosTheta1,cosTheta2)
    indBDen        = bDenominator(cosTheta1,cosTheta2,cosdTheta)
    del cosTheta1, cosTheta2, cosdTheta
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
    np.save(TEMP_DIRECTORY+'plotData_{}.npy'.format(hashlib.md5(str(galaxies).encode('utf-8')).hexdigest()),
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
    

def strload(filename,units):
    """Loads up CF2 files and uses them to rebuild the hash database.
    Returns a list of strings. The strings should be hashed with hashlib.md5(string.encode('utf-8')).hexdigest()"""
    galaxies = common.loadData(filename, dataType = 'CF2')
    if units == 'Mpc/h':
        return str(np.array([(a.x,a.y,a.z,a.v) for a in galaxies]))
    elif units == 'km/s':
        return str(np.array([a.getRedshiftXYZ() + (a.v,) for a in galaxies]))
    else:
        raise ValueError("Value of 'units' must be 'Mpc/h' or 'km/s'. Other unit schemes do not exist at present")


def singleHistogram(positionsStr,xs,intervals,writeOut):
    """Bins individual galaxy-pair data by distance, and writes the results to a np array file."""
    try:
        data =np.load(TEMP_DIRECTORY+'plotData_{}.npy'.format(hashlib.md5(positionsStr.encode('utf-8')).hexdigest()))
        indPsiOneNum = data[0]#['p1n']
        indPsiOneDen = data[1]#['p1d']
        indPsiTwoNum = data[2]#['p2n']
        indPsiTwoDen = data[3]#['p2d']
        indANum = data[4]#['an']
        indADen = data[5]#['ad']
        indBNum = data[6]#['bn']
        indBDen = data[7]#['bd']
        distBetweenG1G2 = data[8]#['dist']
        del data
        print("y",end="",flush=True)
    except FileNotFoundError:
        with np.load(TEMP_DIRECTORY+'plotData_{}.npz'.format(hashlib.md5(positionsStr.encode('utf-8')).hexdigest())) as data: 
        
            indPsiOneNum = data['p1n']
            indPsiOneDen = data['p1d']
            indPsiTwoNum = data['p2n']
            indPsiTwoDen = data['p2d']
            indANum = data['an']
            indADen = data['ad']
            indBNum = data['bn']
            indBDen = data['bd']
            distBetweenG1G2 = data['dist']
            print("z",end="",flush=True)
    #The numpy histogram function returns a tuple of (stuff we want, the bins)
    #Since we already know the bins, we throw them out by taking the [0] element of the tuple.
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
    plt.plot(xs,psipar,'-',label='$\psi_{\parallel}')
    plt.plot(xs,psiprp,'-',label='$\psi_{\perp}')
    plt.title('Velocity correlation')
    plt.xlabel('Distance, {}'.format(units))
    plt.ylabel('Correlation, $(km/s)^2$')
    plt.legend()

    with pdfback.PdfPages(writeOut) as pdf:
        pdf.savefig(fig3)
        pdf.savefig(fig2)
        pdf.savefig(fig)
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)

def standBackStats(inFileList,name,units,writeOut):
    """Do statistics over many input files, for example the three groups of 100 surveys. Average them, plot w/errorbars."""
    print("Actually statting...?")
    pool = Pool()
    allData = np.array(list(pool.map(np.load, inFileList)))
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
        ax5.set_xbound(0,100*100)
        ax6.set_xbound(0,100*100)
    else:
        ax5.set_xbound(0,100)
        ax6.set_xbound(0,100)
    
    with pdfback.PdfPages(writeOut) as pdf:
        pdf.savefig(f)
        
if __name__ == "__main__":
    arrrghs = common.parseCmdArgs([['settings'],
                                   ['-c','--comp'],
                                   ['-H','--hist'],
                                   ['-p','--plots'],
                                   ['-s','--stats']],
                                  ['Settings json file',
                                   'Compute values for individual galaxies',
                                   'Compute histograms (requires a prior or concurrent -c run)',
                                   'Make a plot for every input survey (requires a prior or concurrent -H run)',
                                   'Do the overview stats routine, one plot for all surveys (requires a prior or concurrent -H run)'
                                  ],
                                   [str,'bool','bool','bool','bool'])
    main(arrrghs)
    


