import common
from scipy.spatial import cKDTree
import numpy as np
from multiprocessing import pool
import itertools
import pylab
from numpy.core.umath_tests import inner1d
import matplotlib.backends.backend_pdf as pdfback

#@profile
def main(args):
    """ Compute the velocity correlations on one or many galaxy surveys. 
    """
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

        n,bins = np.histogram(distBetweenG1G2,bins=intervals)
        correlation,bins = np.histogram(distBetweenG1G2,bins=intervals,weights=velocityCorrelation)

        average = correlation/n
        fig = pylab.figure()
        pylab.plot(xs,average)
        pylab.title('3-D velocity correlation')
        pylab.xlabel('Distance, Mpc/h')
        pylab.ylabel('correlation')
        pylab.axis((0,31,0,32))
        pylab.show()
        with pdfback.PdfPages(outfolder+outfile.format(index)) as pdf:
            pdf.savefig(fig)
        pylab.close('all')

class FakeArgs:
    def __init__(self, filename):
        self.settings = filename

if __name__ == "__main__":
    settingsFile = input("Input settings file name: ")
    arrrghs = FakeArgs(settingsFile)
    main(arrrghs)

    
