import common
import numpy as np
import math
import scipy.optimize as optimize
import scipy.spatial as space
import itertools
import os
import multiprocessing
import time
import pylab
import matplotlib.backends.backend_pdf as pdfback
NUM_PROCESSORS = 4
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
    singlerun(filename,outputFile,binsize,chop,override)
        
def genBins(binsize,chop):
    return [x*binsize for x in range(int(chop/binsize)+2)]
    #generates bins for a certain bin size. Stops with the bin that slightly overshoots the chop value
    #Always starts at zero
    
def singlerun(filename,outputFile,binsize,chop,modelOverride=None):
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
    robot = chi_sq_solver(bins,n,selection_function)
    if modelOverride is None:
        #Solve the chi squared optimization for the histogram and selection function
        params = robot.result.x
        #Plot the best fit
        domain = np.arange(0,chop,1)
        model = [selection_function(r,*(robot.result.x)) for r in domain]
        pylab.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*(robot.result.x),chisq = robot.result.fun))
        chisq = robot.result.fun
    else:
        mo = modelOverride["constants"]
        params = [mo['A'], mo['r_0'], mo['n_1'], mo['n_2']]
        chisq = robot.chi_sq(params)
        domain = np.arange(0,chop,1)
        model = [selection_function(r,*params) for r in domain]
        pylab.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*params,chisq = chisq))
        
   
    

    #Add axis labels
    pylab.ylabel("Galaxy count")
    pylab.xlabel("Distance, Mpc/h")
    pylab.title("Distribution of Galaxy Distance")
    pylab.legend()

    fig2 = pylab.figure()
    shellVolume = [common.shellVolCenter(robot.centerbins[i],binsize)  for i in range(len(n))]
    print(shellVolume)
    density = [n[i]/shellVolume[i] for i in range(len(n))]
    pylab.plot(robot.centerbins,density)
    #Save figure
    with pdfback.PdfPages(outputFile+str(binsize)) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    
    #Write paramaters to a file for later use.
    common.writedict(outputFile+str(binsize)+'_params.json',{'constants':{'A':params[0],
                                                                          'r_0':params[1],
                                                                          'n_1':params[2],
                                                                          'n_2':params[3]},
                                                             'info':{'shell_thickness': binsize,
                                                                     'max_radius': chop,
                                                                     'chisq': chisq
                                                                 }
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
    # if value == 0 and r != 0:
    #     print("Uh-oh, the value was zero!")
    #     print(r,A,r_0,n_1,n_2)
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
    outFileName    = settings['survey_output_files']

    if os.path.isdir(hugeFile):
        files = [hugeFile + x for x in os.listdir(hugeFile)]
        densityMap = common.getdict(hugeFile.rstrip('/') + '_fine_density.json')
        density = [densityMap[os.path.basename(f)] for f in files]
    else:
        files = hugeFile
        print("[WARN] Using the huge file will most likely cause problems")

    
    if surveyOverride is not None:
        surveys = surveyOverride
    else:
        surveySeparation = settings['survey_separation_distance']
        numSurveys       = settings['num_surveys']
        surveys = genSurveyPos(surveySeparation, boxSize, numSurveys,hugeFile)
    
    selectionParams = common.getdict(settings['selection_function_json'])
    

    start = time.time()
    pool = multiprocessing.Pool(processes = NUM_PROCESSORS)
    
    listOfSurveyContents = pool.starmap(surveyOneFile,zip(files,
                                                          itertools.repeat(surveys),
                                                          itertools.repeat(selectionParams),
                                                          density))
    print("That took {} seconds.".format(time.time()-start))
    #Format of listOfSurveyContents:
    #List of 1000 elements.
    #Each 'element' is a list of numSurveys elements, each element of which is a list of rows that belong to that
    #survey.

    #e.g. [
    #       [
    #         [rows from survey 1],
    #         [rows from survey 2],
    #         ...],
    #       [],[],...,[]]
    info = []
    surveyContent = transposeMappedSurvey(listOfSurveyContents)
    for i,surveyFinal in enumerate(surveyContent):
        surveyFileName = outFileName + str(i) + '.mil'
        with open(surveyFileName, 'w') as surveyFile:
            for line in surveyFinal:
                surveyFile.write(line)
        info.append({'name':surveyFileName,'center':surveys[i]})
    common.writedict(outFileName + '.json', info)
    common.writedict(outFileName + '_info.json',{'selection_params': selectionParams,
                                            'settings': settings})

def transposeMappedSurvey(data):
    # takes a list of lists of surveys, see above in section "Format of listOfSurveyContents"
    #and flattens it into a single survey
    surveyContent = [[] for i in range(len(data[0]))]
    for mapResult in data:
        for i,survey in enumerate(mapResult):
            for line in survey:
                surveyContent[i].append(line)
    return surveyContent

def surveyOneFile(hugeFile, surveys,selectionParams,density):
    #Note: it's only called a hugeFile because I'm lazy and don't want to change its name.
    rng = np.random.RandomState()
    surveyContent = [[] for i in range(len(surveys))]
    galaxies = common.loadData(hugeFile,dataType = 'millPos')
    #rint(galaxies[0:20])
    galaxPos = [[galaxy[0],galaxy[1],galaxy[2]] for galaxy in galaxies]
    distances = space.distance.cdist(galaxPos,surveys)
    selection_values = selection_function(distances,**(selectionParams["constants"]))
    wantDensity = selection_values / common.shellVolCenter(distances,selectionParams['info']['shell_thickness'])
    probability = wantDensity / density
    dice = rng.random_sample(probability.shape)
    toAdd = dice < probability
    for i,galaxy in enumerate(toAdd):
        for j,addBool in enumerate(galaxy):
            if addBool:
                rawLine = galaxies[i][3]
                surveyContent[j].append(rawLine)
    print("{} complete!".format(hugeFile))
    return surveyContent
                        
# def surveyCheck(info, surveys, params, density):
#     #Deprecated and replaced by numpy
#     #This function should go through all the surveys and determine the distance from the data point (info) to the
#     #center of the survey, then figure out the probability that you should pick the point in question.
#     selection = lambda x: selection_function(x,**(params["constants"]))
#     #selection is the specific version of the generalized selection function, a.k.a. the selection function
#     #with all the constants set to their values.
#     galaxyPos = np.array((info[0],info[1],info[2]))
#     distances = [np.linalg.norm(np.array(r)-galaxyPos) for r in surveys]
#     #The default np linalg norm is the same as sqrt( sum(x**2) ) for x in array
#     surveyAdd = []
#     for d in distances:
#         wantDensity = selection(d) / common.shellVolCenter(d,params['info']['shell_thickness'])
#         probability = wantDensity/density
#         surveyAdd.append(np.random.random_sample() < probability)
        
#         # if addBool:
#     #         print("{: >10,.2e}{: >10,.2e}{: >10,.2e}{: >10,.2e}{: >10,.2e}{: >10}".format(d,
#     #                                                                                       rawSelection,
#     #                                                                                       wantDensity,
#     #                                                                                       probability,
#     #                                                                                       dieroll,
#     #                                                                                       i))
#     return surveyAdd

def genSurveyPos(separation, boxsize, numSurveys,files):
    surveys = [] #list of surveys. Each survey is a tuple (x,y,z) of starting position
    numCatches = 0 #number of times we've tried to grab a position and failed
    millennium = common.MillenniumFiles(files)
    assert numSurveys > 0
    
    if not numSurveys*separation**3 < (boxsize[0] - separation)*(boxsize[1] - separation)*(boxsize[2] - separation):
        #Estimate the volume of the box and the volume of the surveys to determine whether we can physically
        #fit numSurveys into the box. This is a lower bound, so you might get the warning and still be fine.
        print("[WARN] there may be too many surveys for this box size and separation distance.")
    rng = np.random.RandomState()
    for i in range(numSurveys):
        while True:
            #Note: this is basically the equivalent of bogosort as far as algorithm efficiency
            #is concerned. If you try to cram too many surveys into a box it will fail. There's a built in
            #failsafe that detects infinite looping by failing after it tries too many times. (currently 500,000)
            edgeBound = separation/2
            randomCoord = (rng.uniform(edgeBound,boxsize[0]-edgeBound),
                           rng.uniform(edgeBound,boxsize[1]-edgeBound),
                           rng.uniform(edgeBound,boxsize[2]-edgeBound))
            galaxyCoord = millennium.getACloseGalaxy(randomCoord)
            distances = [math.sqrt((r1[0]-r2[0])**2+
                                   (r1[1]-r2[1])**2+
                                   (r1[2]-r2[2])**2)
                             for r1,r2 in zip(itertools.repeat(galaxyCoord),surveys)]
            if all(distance > separation for distance in distances):
                #All is officially the 'coolest function ever.' All of an empty list is true!
                surveys.append(galaxyCoord)
                break
            else:
                numCatches += 1
            if numCatches > 500000:
                raise RuntimeError("We're probably in an infinite loop. Try reducing the number of surveys generated.")
    print("Caught {}!".format(numCatches))
    return surveys



def transpose(args):
    survey_info = common.getdict(args.survey_file)
    for survey in survey_info:
        outCF2String = "" 
        with open(survey['name'],'r') as csvFile:
            for line in csvFile:
                row = line.strip().split(',')
                center = survey['center']
                cf2row = [0,#cz
                          common.distance((row[0],row[1],row[2]),center),#distance (mpc/h)
                          0,#peculiar velocity km/sec
                          0,#dv
                          0,#longitude degrees
                          0]#latitude degrees
                #WARNING: The CF2 conversion algorithm is not complete yet!
                outCF2String = outCF2String + '{}  {}  {}  {}  {}  {}\n'.format(*cf2row)
        with open(survey['name'] + '_cf2.txt', 'w') as cf2outfile:
            cf2outfile.write(outCF2String)
            
if __name__ == "__main__":
    print("This does not run standalone.")
