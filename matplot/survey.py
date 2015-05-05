import common
import numpy as np
import pylab
import math
import scipy.optimize as optimize
import matplotlib.backends.backend_pdf as pdfback
import itertools

np.seterr(all='raise')
#When having problems with dividing by zero, we can debug more easily by having execution
#stop completely when we encounter one, instead of continuing on with only a warning

def statsrun(args):
    all_settings = common.getdict(args.settings)
    binsize = int(all_settings["binsize"])
    outputFile = all_settings["output_filename"]
    filename = all_settings["survey_filename"]
    chop = float(all_settings["chop"])

    singlerun(filename,outputFile,binsize,chop)
        
def genBins(binsize,chop):
    return [x*binsize for x in range(int(chop/binsize)+2)]
    #generates bins for a certain bin size. Stops with the bin that slightly overshoots the chop value
    #Always starts at zero
    
def singlerun(filename,outputFile,binsize,chop):
    fig = pylab.figure()
    galaxies = common.loadData(filename, dataType = "CF2")
    distances = [galaxy.d for galaxy in galaxies]
    #get a list of all the distances to galaxies. This will let us send it directly to the histogram function

    # for i in range(len(distances)-1,-1,-1):
    #     #loop BACKWARDS through the array and get rid of entries bigger than the chop value.
    #     #Why are we doing this? Does it actually affect the results positively?
    #     #Answer: it appears this was an artifact of a previous chop implementation.
    #     if distances[i] > chop:
    #         del distances[i]
    
    bins_orig = genBins(binsize,chop)

    #Make a histogram using pylab histogram function.
    n, bins, patches = pylab.hist(distances, bins_orig, histtype="stepfilled",label="Galaxy Distribution,\n binsize={:.2f}Mpc".format(binsize))

    #Change visual properties of the histogram
    pylab.setp(patches, 'facecolor','g','alpha',0.75)

    #Solve the chi squared optimization for the histogram and selection function
    robot = chi_sq_solver(bins,n,selection_function)

    #Plot the best fit
    domain = np.arange(0,chop,1)
    model = [selection_function(r,*(robot.result.x)) for r in domain]
    pylab.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*(robot.result.x),chisq = robot.result.fun))

    #Add axis labels
    pylab.ylabel("Galaxy count")
    pylab.xlabel("Distance, Mpc/h")
    pylab.title("Distribution of Galaxy Distance")
    pylab.legend()

    fig2 = pylab.figure()
    shellVolume = [((4/3)*math.pi*(robot.centerbins[i])**3-((4/3)*math.pi*(robot.centerbins[i-1])**3 if i > 0 else 0)) for i in range(len(n))]
    print(shellVolume)
    density = [n[i]/shellVolume[i] for i in range(len(n))]
    pylab.plot(robot.centerbins,density)
    #Save figure
    with pdfback.PdfPages(outputFile+str(binsize)) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    params = robot.result.x
    #Write paramaters to a file for later use.
    common.writedict(outputFile+str(binsize)+'_params.json',{'A':params[0],
                                                             'r_0':params[1],
                                                             'n_1':params[2],
                                                             'n_2':params[3]
                                                            })

class chi_sq_solver:
    def __init__(self,bins,ys,function):
        #Initialize object members
        self.bins = bins
        self.ys = ys
        self.centerbins = self.centers(self.bins)
        self.function = function
        #Calculate minimum chi squared fit
        self.result = optimize.minimize(self.chi_sq,
                                        np.array([1000,10,0.005,0.005]),
                                        bounds = [(1,None),(0.01,None),(0,20),(0,20)])
        
    def chi_sq(self, args):
        #Return the chi_squared statistic for the binned data and the arguments for the function
        sum = 0
        for i in range(len(self.ys)-1):
            try:
                E = self.function(self.centerbins[i], *args)
            except FloatingPointError:
                print("Oh no! There was a floating point error.")
                print(self.centerbins[i],*args)
                exit()
            sum += (self.ys[i]-E)**2/E
        return sum

    def centers(self,bins):
        #Calculate the centers of the bins, or the average x value of each bin.
        #Would be mathematically cool if we could average the x values of all the data points,
        #but that probably shouldn't affect anything substantially.
        centers = []
        for i in range(len(bins)-1):
            centers.append((bins[i+1]-bins[i])/2+bins[i])
            #Take the average value of two bins and add it to the lower bin value
        return centers

def selection_function(r, A, r_0, n_1, n_2):
    ratio = r/r_0
    #Selection function as defined by what I got in my email from Professor Feldman
    value = A*(ratio**n_1)*(1+ratio**(n_1+n_2))**-1
    if value == 0 and r != 0:
        print("Uh-oh, the value was zero!")
        print(r,A,r_0,n_1,n_2)
    return value



def selectrun(args):
    #I'll need from args:
    #The input file name (5+gb csv file from millennium)
    #Density of the input file name
    #The survey function (could be the filename of the function parameter json)
    #minimum distance between surveys
    #number of surveys (?)
    #Capitalized variables refer to variables that have not been implemented yet.
    
    #I'll need to go through each line in the file manually to avoid memory overflows.

    #Now: make a list of the survey starting points. The data structure should be:
    #list of origin tuples, (x,y,z). The index will be the ID of the survey. So we save each
    #survey in a file named based on the list index.

    #The survey starting points will all need to be more than MIN_DIST from each other, so I will
    #either need a way of generating random points that are a certain minimum distance from each other or I'll
    #need to manually make a list of a bunch of points to use, then recycle them. The first method would be
    #better, under some circumstances probably.
    settings = common.getdict(args.settings)
    
    hugeFile       = settings["dataset_filename"]
    density        = settings['dataset_density']
    surveyOverride = settings['survey_position_override']
    boxSize        = settings['box_size']
    if surveyOverride is not None:
        surveys = surveyOverride
    else:
        surveySeparation = settings['survey_separation_distance']
        numSurveys       = settings['num_surveys']
        surveys = genSurveyPos(surveySeparation, boxSize, numSurveys)
    
    selectionParams = common.getdict(settings['selection_function_json'])
    with open(FILENAME, 'r') as theFile:
        for i,rawline in enumerate(theFile):
            if i != 0:
                line = rawline.strip()
                row = line.split(',')
                surveyCheck(row, surveys, selectionParams)

def surveyCheck(info, surveys, params):
    #This function should go through all the surveys and determine the distance from the data point (info) to the
    #center of the survey, then figure out the probability that you should pick the point in question.
    selection = lambda x: selection_function(x,**params)
    #selection is the specific version of the generalized selection function, a.k.a. the selection function
    #with all the constants set to their values.
    x = info[0]
    y = info[1]
    z = info[2]
    distances = [sqrt( (x-r[0])**2 +
                       (y-r[1])**2 +
                       (z-r[2])**2 ) for r in surveys]
    

def genSurveyPos(separation, boxsize, numSurveys):
    surveys = []
    numCatches = 0
    assert numSurveys > 0
    if not numSurveys*separation**3 < (boxsize[0] - separation)*(boxsize[1] - separation)*(boxsize[2] - separation):
        print("[WARN] there may be too many surveys for this box size and separation distance.")
    rng = np.random.RandomState()
    for i in range(numSurveys):
        while True:
            #Note: this is basically the equivalent of bogosort as far as algorithm efficiency
            #is concerned. If you try to cram too many surveys into a box it *WILL* keep on running forever.
            #So don't do that. The assert statements should keep a reasonable amount of 
            edgeBound = separation/2
            surveyCoord = (rng.uniform(edgeBound,boxsize[0]-edgeBound),
                           rng.uniform(edgeBound,boxsize[1]-edgeBound),
                           rng.uniform(edgeBound,boxsize[2]-edgeBound))
            distances = [math.sqrt((r1[0]-r2[0])**2+
                                   (r1[1]-r2[1])**2+
                                   (r1[2]-r2[2])**2)
                             for r1,r2 in zip(itertools.repeat(surveyCoord),surveys)]
            if all(distance > separation for distance in distances):
                #All is officially the 'coolest function ever.' All of an empty list is true!
                surveys.append(surveyCoord)
                break
            else:
                numCatches += 1
            if numCatches > 500000:
                print("We tried SO HARD to make this survey for you, but it just didn't work.")
                print("We're very sorry. We might still be able to make you a survey if you reduce")
                print("the number of surveys that we have to put into this box and try again.")
                print("")
                print("Sincerely,")
                print("The elves that sit in your laptop doing hundreds of math problems per second.")
                exit()
    print("Caught {}!".format(numCatches))
    return surveys
    
if __name__ == "__main__":
    print("This does not run standalone.")
