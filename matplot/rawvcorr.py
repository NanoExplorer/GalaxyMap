import common
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import pool
import itertools
import matplotlib.pyplot as plt
from numpy.core.umath_tests import inner1d
import matplotlib.backends.backend_pdf as pdfback

#@profile
def main(args):
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
    print("Incomplete function - see comments")
    exit()
    #Get setup information from the settings file
    settings = common.getdict(args.settings)
    numpoints = settings["numpoints"]
    outfolder = settings["output_data_folder"]
    outfile   = settings["output_file_name"]
    rawInFile = settings["input_file"]
    step_type = settings["step_type"]
    dr =        settings["dr"]
    min_r =     settings["min_r"]
    if settings["many"]:
        #If there are lots of files, set them up accordingly.
        inFileList = [rawInFile.format(x+settings['offset']) for x in range(settings["num_files"])]
    else:
        inFileList = [rawInFile]
    xs,intervals = common.genBins(min_r,numpoints,dr,step_type)
    for index,infile in enumerate(inFileList):
        #Load the survey
        galaxies = np.array(common.loadData(infile, dataType = "millVel"))
        print(galaxies.shape)
        #Put just the galaxy positions into one array
        positions = galaxies[:,0:3] # [(x,y,z),...]
        velocities = galaxies[:,3:6]
        
        kd = cKDTree(positions)
        pairs = kd.query_pairs(max(intervals))
        npPairs = np.array(list(pairs))
        g1pos = positions[npPairs[:,0]]
        g2pos = positions[npPairs[:,1]]

        g1vs = velocities[npPairs[:,0]]
        g2vs = velocities[npPairs[:,1]]

        distBetweenG1G2 = np.linalg.norm(g2pos-g1pos,axis=1)

        velocityCorrelation = inner1d(g1vs,g2vs) / 10**4

        c11 = g1vs[:,0]*g2vs[:,0]
        c12 = g1vs[:,0]*g2vs[:,1]
        c13 = g1vs[:,0]*g2vs[:,2]
        c21 = g1vs[:,1]*g2vs[:,0]
        c22 = g1vs[:,1]*g2vs[:,1]
        c23 = g1vs[:,1]*g2vs[:,2]
        c31 = g1vs[:,2]*g2vs[:,0]
        c32 = g1vs[:,2]*g2vs[:,1]
        c33 = g1vs[:,2]*g2vs[:,2]
        
        n,bins = np.histogram(distBetweenG1G2,bins=intervals)
        
        correlation11,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c11)
        correlation12,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c12)
        correlation13,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c13)
        correlation21,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c21)
        correlation22,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c22)
        correlation23,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c23)
        correlation31,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c31)
        correlation32,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c32)
        correlation33,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=c33)

        a11 = correlation11/n
        a12 = correlation12/n
        a13 = correlation13/n
        a21 = correlation21/n
        a22 = correlation22/n
        a23 = correlation23/n
        a31 = correlation31/n
        a32 = correlation32/n
        a33 = correlation33/n
       
        f, ((ax11,ax12,ax13),
              (ax21,ax22,ax23),
              (ax31,ax32,ax33)) = plt.subplots(3,3,sharex='col',sharey='row',figsize=(11,8.5))
        
        ax11.plot(xs,a11)
        ax12.plot(xs,a12)
        ax13.plot(xs,a13)
        ax21.plot(xs,a21)
        ax22.plot(xs,a22)
        ax23.plot(xs,a23)
        ax31.plot(xs,a31)
        ax32.plot(xs,a32)
        ax33.plot(xs,a33)
        
        #set x axis and y axis to be the same
        #go out to until correlation is zero
        
        f.suptitle('3-D velocity correlation')
        ax31.set_xlabel('Distance, Mpc/h')
        ax32.set_xlabel('Distance, Mpc/h')
        ax33.set_xlabel('Distance, Mpc/h')
        
        ax11.set_ylabel('correlation, $(km/s)^2$')
        ax21.set_ylabel('correlation, $(km/s)^2$')
        ax31.set_ylabel('correlation, $(km/s)^2$')
        
        with pdfback.PdfPages(outfolder+outfile.format(index)) as pdf:
            pdf.savefig(f)
        pylab.close('all')

class FakeArgs:
    def __init__(self, filename):
        self.settings = filename

if __name__ == "__main__":
    settingsFile = input("Input settings file name: ")
    arrrghs = FakeArgs(settingsFile)
    main(arrrghs)

    
