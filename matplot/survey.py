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
    distances = [galaxy.d for galaxy in galaxies]
    #get a list of all the distances to galaxies. This will let us send it directly to the histogram 

    bins_orig = genBins(binsize,chop)

    #Make a histogram using pylab histogram function.
    n, bins, patches = pylab.hist(distances, bins_orig, histtype="stepfilled",label="Galaxy Distribution,\n binsize={:.2f}Mpc".format(binsize))

    #Change visual properties of the histogram
    pylab.setp(patches, 'facecolor','g','alpha',0.75)
    robot = chi_sq_solver(bins,n,selection_function)
    if modelOverride is None:
        #If we don't have an existing model to use, we find a best fit and plot it
        #Solve the chi squared optimization for the histogram and selection function
        params = robot.result.x
        #Plot the best fit
        domain = np.arange(0,chop,1)
        model = [selection_function(r,*(robot.result.x)) for r in domain]
        pylab.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*(robot.result.x),chisq = robot.result.fun))
        chisq = robot.result.fun
    else:
        #Plot the model given in the settings function instead of calculating a new one
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
    pylab.axis([0,chop,0,700])
    fig2 = pylab.figure()
    shellVolume = [common.shellVolCenter(robot.centerbins[i],binsize)  for i in range(len(n))]
    pylab.title("Galaxies per Cubic Mpc")
    pylab.xlabel("Distance, Mpc/h")
    pylab.ylabel("Density, galaxies/(Mpc/h)^3")
    density = [n[i]/shellVolume[i] for i in range(len(n))]
    pylab.plot(robot.centerbins,density)
    #Save figure
    with pdfback.PdfPages(outputFile+str(binsize)) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    if modelOverride is None:
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
    pylab.close('all')

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
    #The input file folder name (5+gb csv file from millennium)
    #Density of the input file name (in shells around the survey centers)
    #The survey function (the filename of the function parameter json)
    #minimum distance between surveys
    #number of surveys
    
    #I'll need to go through the database in blocks using the usual sub-box method 

    settings = common.getdict(args.settings)
    
    hugeFile       = settings["dataset_filename"]
    #density        = settings['dataset_density']
    surveyOverride = settings['survey_position_override']
    boxSize        = settings['box_size']
    outFileName    = settings['survey_output_files']

    if os.path.isdir(hugeFile):
        files = [hugeFile + x for x in os.listdir(hugeFile)]
    else:
        files = hugeFile
        print("[WARN] Using the gigantic file is no longer supported and will probably cause really weird errors.")

    #Now: make a list of the survey starting points. The data structure should be:
    #list of origin tuples, (x,y,z). The index will be the ID of the survey. So we save each
    #survey in a file named based on the list index.
    
    if surveyOverride is not None:
        surveys = surveyOverride
    else:
        surveySeparation = settings['survey_separation_distance']
        numSurveys       = settings['num_surveys']
        surveys = genSurveyPos(surveySeparation, boxSize, numSurveys,hugeFile)

    #Generate a coordinate system for each survey to use
    #Method: Since the normal distribution is symmetric around its mean, if each coordinate is
    #normally distributed around a mean of zero then the distribution is spherically symmetric
    #While the variable name is "up vector," it more closely represents the rotation angle to
    #rotate the coordinate system around. 
    rawSurveyUpVectors = np.random.normal(0,0.1,(len(surveys),3))
    upVectorLengths = np.linalg.norm(rawSurveyUpVectors,axis=1) #The axis=1 means this is an array of 3-vectors,
                                                                #and not a single 5x3 vector

    #Normalized version of the original 'raw' survey up vectors
    surveyUpVectors = np.array([vec/length for vec,length in zip(rawSurveyUpVectors,list(upVectorLengths))])
    surveyRotationAngles = np.random.uniform(0,2*math.pi,surveyUpVectors.shape[0])
    #Rotation angles is the angle by which the coordinate system is rotated about the 'up' vector

    #Now we make a rotation matrix out of the vector-angle rotation
    #Definition from http://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis-angle
    axisAngletoMatrix = lambda r, theta: (math.cos(theta)*np.identity(3) +
                                          math.sin(theta)*common.crossProductMatrix(r) +
                                          (1-math.cos(theta))*np.outer(r,r))
    #It turns out that the rows of these rotation matrices form the basis vectors for the new coordinate system
    rotationMatrices = [axisAngletoMatrix(r,theta) for r,theta in zip(surveyUpVectors,list(surveyRotationAngles))]
    
    selectionParams = common.getdict(settings['selection_function_json'])

    #Grab the pre-computed distance files if they exist, or if not generate them.
    distFileBase = outFileName + "_{:x}/".format(hash(tuple([tuple(x) for x in surveys]))) 
    distanceFiles = [distFileBase + os.path.basename(os.path.splitext(milFile)[0]) + '.npy' for milFile in files]
    #Distance file format: outFile location + hash of survey centerpoints / xi.yi.zi.npy
    
    pool = multiprocessing.Pool(processes = NUM_PROCESSORS)

    if not os.path.exists(distFileBase):
        start = time.time()
        print("Generating distance data...")
        os.mkdir(distFileBase)
        pool.starmap(distanceOneBox,zip(files,
                                        itertools.repeat(surveys),
                                        distanceFiles))
        print("Generating distance data took {} seconds.".format(time.time()-start))
    else:
        print("Found distance data!")

    #Generate lookup-tables for 'original-number-of-galaxies' if they don't already exist
    boxMaxDistance = space.distance.euclidean([0,0,0],boxSize)
    if not os.path.exists(distFileBase+'hist.npy'):
        print("Generating histograms...")
        start = time.time()
        
        listOfHistograms = pool.starmap(surveyBins,zip(distanceFiles,
                                                       itertools.repeat(selectionParams["info"]["shell_thickness"]),
                                                       itertools.repeat(boxMaxDistance)))
        full_histogram = sum(listOfHistograms)
        np.save(distFileBase+'hist.npy',full_histogram)
        print("Generating histograms took {} seconds.".format(time.time()-start))
        #because the surveyBins function returns a numpy array, the sum function will add them all together element-wise!
    else:
        print("Found histogram!")
        full_histogram = np.load(distFileBase+'hist.npy')
        
    print("Generating surveys...")
    start = time.time()
    listOfSurveyContents = pool.starmap(surveyOneFile,zip(files,
                                                          distanceFiles,
                                                          itertools.repeat(selectionParams),
                                                          itertools.repeat(full_histogram),
                                                          itertools.repeat(boxMaxDistance)
                                                      ))
    print("Generating surveys took {} seconds.".format(time.time()-start))
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
    #Jam the arrays back together 
    surveyContent = transposeMappedSurvey(listOfSurveyContents)
    #write them to the disk
    for i,surveyFinal in enumerate(surveyContent):
        surveyFileName = outFileName + str(i) + '.mil'
        with open(surveyFileName, 'w') as surveyFile:
            for line in surveyFinal:
                surveyFile.write(line)
        info.append({'name':surveyFileName,'center':surveys[i],'rot':[[d for d in c] for c in rotationMatrices[i]]})
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


def distanceOneBox(hugeFile,surveys,outFile):
    #Generate distance data for one sub-box - distance from each galaxy to each survey center
    #These distances are not returned, instead they are only written to the disk.
    galaxies = common.loadData(hugeFile,dataType = 'millPos')
    galaxPos = [galaxy[0:3] for galaxy in galaxies]
    distances = space.distance.cdist(galaxPos,surveys)
    np.save(outFile,distances)
    #Write distances to files for later use
    #Also this algorithm seems VERY fast. It also doesn't seem to get slower with more surveys!
    

def surveyBins(distanceFile,binsize,boxMaxDistance):
    """
    Takes in the survey centerpoints and calculates bins for each of the surveys
    By that I mean it counts the number of galaxies in each bin up to 'chop' using the same method
    as surveystats.

    The distance file MUST exist before this method is called.
    """
    distances = np.load(distanceFile)
    numSurveys = distances.shape[1]
    bins = genBins(binsize,boxMaxDistance)
    histogram = [] #np.zeros((distances.shape[1],int(boxMaxDistance/binsize)+1))
    #list of surveys. Each survey will contain a list of bins,
    #Each bin will contain the number of galaxies in that bin.
    #This is the list we're generating.

    #This method is not necessarily the best - it basically counts galaxies.
    #Maybe there's a better way?
    #Answer - definitely! Numpy.histogram.
    for i in range(numSurveys):
        thisSurveyDistances = distances[:,i] #This means take distances for all the galaxies, but only the ith survey
        #That works since the first dimension of distances is the galaxies and the second dimension is surveys
        hist,edges = np.histogram(thisSurveyDistances, bins)
        histogram.append(hist)
    # for i,galaxy in enumerate(distances):
    #     for surveyNum,distance in enumerate(galaxy):
    #         histogram[surveyNum][int(distance/binsize)] += 1
    return np.array(histogram)
    
def surveyOneFile(hugeFile,distanceFile,selectionParams,histogram,boxMaxDistance):
    """
    Given the original data, distances, wanted numbers, and other parameters we actually generate the
    mock surveys. This is currently the biggest bottleneck of the program, but I'm ot entirely sure why.
    """
    #Set up variables
    rng = np.random.RandomState() #Make a process-safe random number generator
    distances = np.load(distanceFile) #load the distance file
    surveyContent = [[] for i in range(distances.shape[1])] #Make the skeleton structure for the end result
    galaxies = common.loadData(hugeFile,dataType = 'millRaw') #Load the galaxies
    binsize = selectionParams['info']['shell_thickness'] #Load the size of a bin
    bins = genBins(binsize,boxMaxDistance)
    numSurveys = distances.shape[1]
    
    #Do calculations
    selection_values = selection_function(distances,**(selectionParams["constants"]))
    wantDensity = selection_values / common.shellVolCenter(distances,binsize)
    distBin = [np.digitize(distances[:,n],bins)-1 for n in range(numSurveys)]
    originalCount = np.transpose(np.array([histogram[n][distBin[n]] for n in range(numSurveys)]))
    volumes = common.shellVolCenter(np.transpose(np.array(distBin))*binsize + (binsize/2),binsize)
    originalDensity = originalCount / volumes
    #I'm currently under the impression that this for loop is the main bottleneck in this function.
    #It isn't a complicated task, so it might be worthwhile to consider alternate implementations.
    #Let's see...
    #This for loop uses information from distances, binsize, histogram
    
    # for i,galaxy in enumerate(distances):
    #     for j,surveyDist in enumerate(galaxy):
    #         distBin = int(distBinNotAnInt[i][j])
    #         originalDensity[i][j] = histogram[j][distBin] / common.shellVolCenter(distBin*binsize + (binsize/2),binsize)
    probability = wantDensity / originalDensity
    dice = rng.random_sample(probability.shape)
    toAdd = dice < probability
    # for i,galaxy in enumerate(toAdd):
    #     for j,addBool in enumerate(galaxy):
    #         if addBool:
    #             rawLine = galaxies[i][3]
    #             surveyContent[j].append(rawLine)
    surveyContent = [np.array(galaxies)[toAdd[:,n]] for n in range(numSurveys)]
    return surveyContent




def genSurveyPos(separation, boxsize, numSurveys,files):

    #The survey starting points will all need to be more than MIN_DIST from each other,
    #To achieve this, and make sure that surveys start AT galaxies, I'll select
    #random galaxies from the simulation then make sure that they are far enough from each other
    
    surveys = [] #list of surveys. Each survey is a tuple (x,y,z) of starting position
    numCatches = 0 #number of times we've tried to grab a position and failed
    millennium = common.MillenniumFiles(files)
    assert numSurveys > 0
    
    if not numSurveys*separation**3 < (boxsize[0] - 2*separation)*(boxsize[1] - 2*separation)*(boxsize[2] - 2*separation):
        #Estimate the volume of the box and the volume of the surveys to determine whether we can physically
        #fit numSurveys into the box. This is a lower bound, so you might get the warning and still be fine.
        print("[WARN] there may be too many surveys for this box size and separation distance.")
    #rng = np.random.RandomState()
    for i in range(numSurveys):
        while True:
            #Note: this is basically the equivalent of bogosort as far as algorithm efficiency
            #is concerned. If you try to cram too many surveys into a box it will fail. There's a built in
            #failsafe that detects infinite looping by failing after it tries too many times. (currently 10000)
            galaxy = millennium.getARandomGalaxy()
            galaxyCoord = (galaxy.x,galaxy.y,galaxy.z)
            distances = [math.sqrt((r1[0]-r2[0])**2+
                                   (r1[1]-r2[1])**2+
                                   (r1[2]-r2[2])**2)
                             for r1,r2 in zip(itertools.repeat(galaxyCoord),surveys)]
            for i,c in enumerate(galaxyCoord):
                distances.append(c)
                distances.append(boxsize[i]-c)
            if all(distance > separation for distance in distances):
                #All is officially the 'coolest function ever.' All of an empty list is true!
                surveys.append(galaxyCoord)
                break
            else:
                numCatches += 1
            if numCatches % 100 == 0:
                print("So far we have tried {} times".format(numCatches))
            if numCatches > 10000:
                raise RuntimeError("We're probably in an infinite loop. Try reducing the number of surveys to make.")
    print("Caught {}!".format(numCatches))
    print(surveys)
    return surveys



def transpose(args):
    survey_info = common.getdict(args.survey_file)
    for survey in survey_info:
        outCF2String = "" 
        with open(survey['name'],'r') as csvFile:
            for line in csvFile:
                galaxy=common.MillenniumGalaxy(line)
                center = survey['center']
                rotationMatrix = np.matrix(survey['rot'])
                ontoGalaxy = np.array([galaxy.x-center[0],galaxy.y-center[1],galaxy.z-center[2]])
                #ontoGalaxy is the vector from the survey origin to the galaxy
                rotatedCoord = ontoGalaxy * rotationMatrix
                x = rotatedCoord.item(0)
                y = rotatedCoord.item(1)
                z = rotatedCoord.item(2)
                rho = space.distance.euclidean(ontoGalaxy,[0,0,0])
                phi = math.acos(z/rho)*180/math.pi - 90
                theta = math.atan2(y,x)*180/math.pi + 180
                peculiarVel = np.dot(ontoGalaxy,[galaxy.velX,galaxy.velY,galaxy.velZ])/rho
                #posVec = ontoGalaxy/space.distance.euclidean(ontoGalaxy,(0,0,0))
                cf2row = [0,#cz
                          rho,#distance (mpc/h)
                          peculiarVel,#peculiar velocity km/sec
                          0,#dv
                          theta,#longitude degrees - 0 - 360
                          phi]#latitude degrees - -90 - 90
                #Still missing redshift calculation, but that's it!
                outCF2String = outCF2String + '{}  {}  {}  {}  {}  {}\n'.format(*cf2row)
        with open(survey['name'] + '_cf2.txt', 'w') as cf2outfile:
            cf2outfile.write(outCF2String)
            
if __name__ == "__main__":
    print("This does not run standalone.")
    
