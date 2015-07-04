import common
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import Pool
import itertools
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdfback
#import pdb
from numpy.core.umath_tests import inner1d #Note: this function has no documentation and could easily be deprecated.
#if that happens, you can always use the syntax (a*b).sum(axis=1), which is ever so slightly slower and much more
#ugly.
#Alternate definition:
# def inner1d(a,b):
#     return (a*b).sum(axis=1)

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
    #NOTE: If this line gives you trouble (e.g. 'pool doesn't have a member called starmap'), just replace
    #that line with this one:
    #plots = pool.map(singlePlotStar,zip(inFileList,
    
    for n,plot in enumerate(plots):
        psione = [x for x in plot[0]]
        psitwo = [y for y in plot[1]]
        a = [z for z in plot[2]]
        b = [dataPoint for dataPoint in plot[3]]
        psiParallel = [thing for thing in plot[4]]
        psiPerpindicular = [pp for pp in plot[5]]
        common.writedict(outfolder+outfile.format(n+settings['offset'])+'_rawdata.json',{'psione':psione,
                                                                                         'psitwo':psitwo,
                                                                                         'a':a,
                                                                                         'b':b,
                                                                                         'xs':xs,
                                                                                         'psi_parallel':psiParallel,
                                                                                         'psi_perpindicular':psiPerpindicular
                                                                                     })
        np.save(outfolder+outfile.format(n+settings['offset'])+'_rawdata.npy',np.array([psione,psitwo,a,b,psiParallel,psiPerpindicular]))
        #End for loop
    if not hasattr(args,'plot'):
        #This function was called by the galaxy.py interface, and we should plot.
        #If the attribute exists, then the if __name__=='__main__': thingy will take care of everything.
        stats(args)

def singlePlotStar(args):
    return singlePlot(*args)
    
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

    #Find zero-distance pairs
    removePairs = kd.query_pairs(np.finfo(float).eps)
    pairs.difference_update(removePairs)
    try:
        data = correlation(pairs,galaxyXYZV,intervals)
    except RuntimeError:
        print("Runtime Error encountered at {}.".format(infile))
        raise
    return data

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
    indPsiOneNum   = psiOneNumerator(g1vs,g2vs,cosdTheta)
    indPsiOneDen   = psiOneDenominator(cosdTheta)
    indPsiTwoNum   = psiTwoNumerator(g1vs,g2vs,cosTheta1,cosTheta2)
    indPsiTwoDen   = psiTwoDenominator(cosTheta1,cosTheta2,cosdTheta)
    #Original A calculations removed in favor of new aFeldman calculations
    #indANum        = aNumerator(cosdTheta,g1dist,g2dist,distBetweenG1G2)
    #indADen        = aDenominator(cosdTheta,distBetweenG1G2)
    indAFeldmanNum = aFeldmanNumerator(cosTheta1,cosTheta2,cosdTheta)
    indAFeldmanDen = aFeldmanDenominator(cosdTheta)
    indBNum        = bNumerator(cosTheta1,cosTheta2)
    indBDen        = bDenominator(cosTheta1,cosTheta2,cosdTheta)

    #The numpy histogram function returns a tuple of (stuff we want, the bins)
    #Since we already know the bins, we throw them out by taking the [0] element of the tuple.
    psiOneNum = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiOneNum)[0]
    psiOneDen = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiOneDen)[0]
    psiTwoNum = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiTwoNum)[0]
    psiTwoDen = np.histogram(distBetweenG1G2,bins = intervals,weights = indPsiTwoDen)[0]
    aNum      = np.histogram(distBetweenG1G2,bins = intervals,weights = indAFeldmanNum)[0]
    aDen      = np.histogram(distBetweenG1G2,bins = intervals,weights = indAFeldmanDen)[0]
    #afNum     = np.histogram(distBetweenG1G2,bins = intervals,weights = indAFeldmanNum)[0]
    #afDen     = np.histogram(distBetweenG1G2,bins = intervals,weights = indAFeldmanDen)[0]
    bNum      = np.histogram(distBetweenG1G2,bins = intervals,weights = indBNum)[0]
    bDen      = np.histogram(distBetweenG1G2,bins = intervals,weights = indBDen)[0]
    
    psione = psiOneNum/psiOneDen
    psitwo = psiTwoNum/psiTwoDen
    a = aNum/aDen
    b = bNum/bDen
    #af = afNum/afDen
    aminusb = (a-b)
    psiParallel = ((1-b)*psione-(1-a)*psitwo)/aminusb
    psiPerpindicular = (a*psitwo-b*psione)/aminusb
    """ This is a pre-made NaN debugging scheme. I'm leaving it in in case it's useful in the future."""
    # for i,num in enumerate(distBetweenG1G2):
    #     if abs(num) < 0.0000001:
    #         print('********** BEGIN UH-OH REPORT *********** ')
    #         print(galaxyPairs[i])
    #         print(lGalaxies[i])
    #         print(rGalaxies[i])
    #         print('********** END UH-OH REPORT ************* ')
    #         raise RuntimeError("TESTING and trapping nans")
    # if np.isnan(np.sum(psione)):
    #     #if there is a nan in the array, the sum of the array will be nan as well.
    #     print('psi 1 is NaN')
    #     print(np.isnan(np.sum(indPsiOneNum)))
    #     print(np.isnan(np.sum(indPsiOneDen)))
    # if np.isnan(np.sum(psitwo)):
    #     print('psi 2 is NaN')
    #     print(np.isnan(np.sum(indPsiTwoNum)))
    #     print(np.isnan(np.sum(indPsiTwoDen)))
    # if np.isnan(np.sum(a)):
    #     print('a is NaN')
    #     print(np.isnan(np.sum(indANum)))
    #     print(np.isnan(np.sum(indADen)))
    # if np.isnan(np.sum(b)):
    #     print('b is NaN')
    #     print(np.isnan(np.sum(indBNum)))
    #     print(np.isnan(np.sum(indBDen)))
    # if np.isnan(np.sum(af)):
    #     print('a feldman is NaN')
    #     print(np.isnan(np.sum(indAFeldmanNum)))
    #     print(np.isnan(np.sum(indAFeldmanDen)))
    """END PRE-MADE NaN DEBUGGING SCHEME ---------------------------------------------------------"""
    return (psione,psitwo,a,b,psiParallel,psiPerpindicular)

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
    
"""These functions are more messy and stuff than the aFeldman functions, but produce the same
results. """
# def aNumerator(cosdTheta,g1d,g2d,r):
#     return (g1d*g2d*((cosdTheta**2)-1)+(r**2)*cosdTheta)*cosdTheta
# def aDenominator(cosdTheta,r):
#     return (cosdTheta**2)*(r**2)

def aFeldmanNumerator(costheta1,costheta2,cosdTheta):
    return costheta1*costheta2*cosdTheta

def aFeldmanDenominator(cosdTheta):
    return cosdTheta**2

def bNumerator(costheta1,costheta2):
    return (costheta1**2)*(costheta2**2)
    
def bDenominator(costheta1,costheta2,cosdTheta):
    return costheta1*costheta2*cosdTheta
    
#@profile
# def kd_query(positions,interval):
#     """Get all pairs of galaxies at distances inside the interval. 
#     interval is a tuple of (upper,lower) bounds for the distance bin.
#     """
#     kd = cKDTree(positions)
#     upper = kd.query_pairs(interval[0])
#     lower = kd.query_pairs(interval[1])
#     upper.difference_update(lower) #Get rid of pairs at separations less than the lower bound
#     # Use this instead of one - two because of time complexity.
#     #see: https://wiki.python.org/moin/TimeComplexity
    
#     return upper

#@profile
def stats(args):
    """Make plots of the data output by the main function"""
    #Get settings
    settings = common.getdict(args.settings)
    outfolder = settings["output_data_folder"]
    outfile   = settings["output_file_name"]
    #XSrawInFile = settings["input_file"]
    
    if settings["many"]:
        inFileList = [outfolder+outfile.format(x+settings['offset'])+'_rawdata.json' for x in range(settings["num_files"])]
    else:
        inFileList = [outfolder+outfile+'_rawdata.json']

    for n,infile in enumerate(inFileList):
        data = common.getdict(infile)
        xs = data['xs']
        a = data['a']
        b = data['b']
        psione = [x/10**4 for x in data['psione']]
        psitwo = [x/10**4 for x in data['psitwo']]

        psipar = data['psi_parallel']
        psiprp = data['psi_perpindicular']
        
        fig = plt.figure()
        plt.plot(xs,a,'-',label="$\cal A$ (Borgani)")
        plt.plot(xs,b,'k--',label="$\cal B$")
        plt.title("Moment of the selection function")
        plt.ylabel("Value (unitless)")
        plt.xlabel("Distance, Mpc/h")
        plt.legend(loc=2)
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.axis((0,31,.62,.815))


        fig2 = plt.figure()
        plt.plot(xs,psione,'-',label="$\psi_1$")
        plt.plot(xs,psitwo,'k--',label="$\psi_2$")
        plt.title("Velocity correlation function")
        plt.xlabel("Distance, Mpc/h")
        plt.ylabel("Correlation, $10^4 (km/s)^2$")
        #plt.axis((0,31,0,32))
        plt.legend()

        fig3 = plt.figure()
        plt.plot(xs,psipar,'-',label='$\psi_{\parallel}')
        plt.plot(xs,psiprp,'-',label='$\psi_{\perp}')
        plt.title("Velocity correlation")
        plt.xlabel("Distance, Mpc/h")
        plt.ylabel("Correlation, $(km/s)^2$")
        plt.legend()

        with pdfback.PdfPages(outfolder+outfile.format(n+settings['offset'])) as pdf:
            pdf.savefig(fig3)
            pdf.savefig(fig2)
            pdf.savefig(fig)
        plt.close('all')

def standBackStats(args):
    """Do statistics over many input files, for example the three groups of 100 surveys. Average them, plot w/errorbars."""
    #Get settings
    settings = common.getdict(args.settings)
    outfolder = settings["output_data_folder"]
    outfile   = settings["output_file_name"]
    #XSrawInFile = settings["input_file"]

    xs = common.getdict(outfolder+outfile.format(settings['offset'])+'_rawdata.json')['xs']
    if settings["many"]:
        inFileList = [outfolder+outfile.format(x+settings['offset'])+'_rawdata.npy' for x in range(settings["num_files"])]
    else:
        raise RuntimeError("The averaging routines require multiple files to average.")
    
    allData = np.array(list(map(np.load, inFileList)))
    #One inFile contains the following: [p1, p2, a, b, psiparallel, psiperpindicular]
    print(allData.shape)
    std = np.std(allData,axis=0)
    avg = np.mean(allData,axis=0)

    correlationScale = (0,30,0,160000)
    momentScale = (0,30,0.25,1.1)
    plotName = "CF2 Group"

    fig1 = plt.figure()
    plt.errorbar(xs, avg[0], yerr=std[0], fmt = '-')
    plt.title('$\psi_1$ Correlation for the {} Survey'.format(plotName))
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Correlation, $(km/s)^2$')
    plt.axis(correlationScale)
    
    fig2 = plt.figure()
    plt.errorbar(xs, avg[1], yerr=std[1], fmt = '-')
    plt.title('$\psi_2$ Correlation for the {} Survey'.format(plotName))
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Correlation, $(km/s)^2$')
    plt.axis(correlationScale)
    
    fig3 = plt.figure()
    plt.errorbar(xs, avg[2], yerr=std[2], fmt = '-')
    plt.title('Moment of the Selection Function, $\cal A$, {} Survey'.format(plotName))
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Value (unitless)')
    plt.axis(momentScale)
    
    fig4 = plt.figure()
    plt.errorbar(xs, avg[3], yerr=std[3], fmt = '-')
    plt.title('Moment of the Selection Function, $\cal B$, {} Survey'.format(plotName))
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Value (unitless)')
    plt.axis(momentScale)

    fig5 = plt.figure()
    plt.errorbar(xs, avg[4], yerr=std[4], fmt = '-')
    plt.title('$\Psi_{{\parallel}}$ Correlation for the {} Survey'.format(plotName))
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Correlation, $(km/s)^2$')
    plt.axis(correlationScale)
    
    fig6 = plt.figure()
    plt.errorbar(xs, avg[5], yerr=std[5], fmt = '-')
    plt.title('$\Psi_{{\perp}}$ Correlation for the {} Survey'.format(plotName))
    plt.xlabel('Distance, Mpc/h')
    plt.ylabel('Correlation, $(km/s)^2$')
    plt.axis(correlationScale)
    
    with pdfback.PdfPages(outfolder+outfile.format("MACRO")) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        pdf.savefig(fig4)
        pdf.savefig(fig5)
        pdf.savefig(fig6)
    
if __name__ == "__main__":
    arrrghs = common.parseCmdArgs([['settings'],
                                   ['-c','--comp'],
                                   ['-p','--plot'],
                                   ['-s','--stats'],
                                   ['-o','--onlystats']],
                                  ['Settings json file',
                                   'only do computations, no plotting',
                                   'only plot, no computations',
                                   'do the overview stats routine after computing',
                                   'do only the overview stats routine'
                                  ],
                                   [str,'bool','bool','bool','bool'])
    if not arrrghs.plot and not arrrghs.onlystats:
        print("computing...")
        main(arrrghs)
    if not arrrghs.comp and not arrrghs.onlystats:
        print('plotting...')
        stats(arrrghs)
    if arrrghs.stats or arrrghs.onlystats:
        print('statting..?')
        print('no, that doesn\'t sound right')
        print('computing statistics...')
        standBackStats(arrrghs)

    


