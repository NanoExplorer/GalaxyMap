import common
import numpy as np
import pylab
import matplotlib.backends.backend_pdf as pdfback
np.seterr(all='raise')
#When having problems with dividing by zero, we can debug more easily by having execution
#stop completely when we encounter one, instead of continuing on with only a warning


def statsrun(args):
    all_settings = common.getdict(args.settings)
    binsize = int(all_settings["binsize"])
    outputFile = all_settings["output_filename"]
    filename = all_settings["survey_filename"]
    chop = float(all_settings["chop"])
    if all_settings["model_override"] is not None:
        override = common.getdict(all_settings["model_override"])
    else:
        override = None

    if "many" in all_settings and all_settings["many"] == True:
        num_files = all_settings["num_files"]
        for x in range(num_files):
            singlerun(filename.format(x),
                      outputFile.format(x),
                      binsize,
                      chop,
                      override
            )
    else:
        singlerun(filename,outputFile,binsize,chop,override)
        
def genBins(binsize,chop):
    return [x*binsize for x in range(int(chop/binsize)+2)]
    #generates bins for a certain bin size. Stops with the bin that slightly overshoots the chop value
    #Always starts at zero
    
def singlerun(filename,outputFile,binsize,chop,modelOverride=None):
    fig = pylab.figure()
    galaxies = common.loadData(filename, dataType = "CF2")
    velocities = [galaxy.v for galaxy in galaxies]
    #get a list of all the galaxies' velocities. This will let us send it directly to the histogram 

    bins_orig = genBins(binsize,chop)

    #Make a histogram using pylab histogram function.
    n, bins, patches = pylab.hist(velocities, bins_orig, histtype="stepfilled",label="Galaxy Distribution,\n binsize={:.2f}Mpc".format(binsize))

    #Change visual properties of the histogram
    pylab.setp(patches, 'facecolor','g','alpha',0.75)    

    #Add axis labels
    pylab.ylabel("Galaxy count")
    pylab.xlabel("Radial Velocity, km/s")
    pylab.title("Distribution of Galaxy radial velocities")
    pylab.axis([0,chop,0,1000])
    
    with pdfback.PdfPages(outputFile+str(binsize)) as pdf:
        pdf.savefig(fig)
    pylab.show()
    pylab.close('all')
            
if __name__ == "__main__":
    args = common.parseCmdArgs([['settings']],['Settings json file'],[str])
    statsrun(args)
    

