import common
import numpy as np
import math
import scipy.spatial as space
import itertools
import os
import multiprocessing
import time
import random
USE_GPU = False
try:
    import pycuda.autoinit
    from pycuda import gpuarray
    print("[INFO] GPU available for use.")
    USE_GPU=True    
except:
    print("[INFO] GPU unavailable.")
    USE_GPU=False


NUM_PROCESSORS = 12

EDGES = [np.array((x,y,z)) for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1]]

def genBins(binsize,chop):
    return [x*binsize for x in range(int(chop/binsize)+2)]
    #generates bins for a certain bin size. Stops with the bin that slightly overshoots the chop value
    #Always starts at zero
    
#@profile
def compute_want_density(r,binsize, A, r_0, n_1, n_2):
    #Warning: Doesn't actually compute a density anymore. 
    if USE_GPU: #Use gpu just changes type of np arrays so that it runs on GPU if available.
        r_in = r.astype(np.float32)
        r_gpu = gpuarray.to_gpu(r_in)
    else:
        r_in = r
        r_gpu = r


    r_calc = (A*((r_gpu/r_0)**n_1)*(1+(r_gpu/r_0)**(n_1+n_2))**-1)
    #(A*(r/r0)^n_1)*(1+(r/r0)^(n1+n2))^-1 is the selection function, i.e. number of galaxies in a binsize Mpc bin
    #centered on r distance from center

    
    if USE_GPU:
        r_out = r_calc.get()
    else:
        r_out = r_calc
  
    return r_out
    
def selectrun(args):
    #I'll need from args:
    #The input file folder name (5+gb csv file from millennium)
    #Density of the input file name (in shells around the survey centers)
    #The survey function (the filename of the function parameter json)
    #minimum distance between surveys
    #number of surveys
    global USE_GPU
    #I'll need to go through the database in blocks using the usual sub-box method 
    USE_GPU = args.gpu and USE_GPU
    #USE_GPU represents whether we can and want to use the gpu. 
    settings = common.getdict(args.settings)
    
    hugeFile       = settings["dataset_filename"]
    #density        = settings['dataset_density']
    surveyOverride = settings['survey_position_override']
    boxSize        = settings['box_size']
    if boxSize[0] != 500 or boxSize[1] != 500 or boxSize[2] != 500:
        print("Not designed with other simulation boxes in mind.")
        exit()
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

    #!!NOTE: The following code generates a random rotation matrix. In the past I had used that to make
    #        A random coordinate system for each survey. All I do with that rotation matrix in this file
    #        is save it to the disk. That information is used to rotate the coords in surveytranspose.py
    #        SurveyTranspose.py is currently configured to IGNORE this. So there are no rotations being done.
        
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

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!END unused code.    

    #Grab the selection function parameters from the file generated by surveystats
    selectionParams = common.getdict(settings['selection_function_json'])

    #Grab the pre-computed distance files if they exist, or if not generate them.
    distFileBase = hugeFile.rstrip('/') + "_distances_{:x}/".format(hash(tuple([tuple(x) for x in surveys]))) 
    distanceFiles = [distFileBase + os.path.basename(os.path.splitext(milFile)[0]) for milFile in files]
    #Distance file format: outFile location + hash of survey centerpoints / xi.yi.zi.npy


    
    pool = multiprocessing.Pool(processes = NUM_PROCESSORS)


    #Generate lookup-tables for 'original-number-of-galaxies' if they don't already exist
    boxMaxDistance = 350 # space.distance.euclidean([0,0,0],boxSize)
    print("Assuming chop distance of 350 Mpc/h")
    if not os.path.exists(distFileBase+'hist{}.npy'.format(selectionParams["info"]["shell_thickness"])):
        print("Generating histograms...")
        start = time.time()
        
        #A map function will always take the function, in this case distsurvey, and apply it to the arguments.
        #for example starmap(*,[(1,1),(2,4),(2,1),(99,10)]) = [1,8,2,990].

        #This does the function distsurvey on every file in the list of files. the repeat function just sends
        #the same thing to every function call.
        #So this will call
        #    distsurvey(file[1], surveys, selectionParams, maxdistance)
        #    distsurvey(file[2], surveys, selectionParams, maxdistance)
        #    distsurvey(file[3], surveys, selectionParams, maxdistance)
        listOfHistograms = pool.starmap(distsurvey,zip(files,
                                                       itertools.repeat(surveys),
                                                       itertools.repeat(selectionParams["info"]["shell_thickness"]),
                                                       itertools.repeat(boxMaxDistance)))
        full_histogram = sum(listOfHistograms)
        #because the surveyBins function returns a numpy array, the sum function will add them all together element-wise!
        #The listOfHistograms is the number of galaxies per bin per box. The sum combines them into one histogram
        #so that it's per bin and not per box.
        
        np.save(distFileBase+'hist{}.npy'.format(selectionParams["info"]["shell_thickness"]),full_histogram)
        print("Generating histograms took {} seconds.".format(time.time()-start))
        
    else:
        print("Found histogram!")
        full_histogram = np.load(distFileBase+'hist{}.npy'.format(selectionParams["info"]["shell_thickness"]))
        
    print("Generating surveys...")
    start = time.time()
    if USE_GPU:
        pool.close()
        #See notes on starmap above in the histogram section
        listOfSurveyContents = itertools.starmap(surveyOneFile,zip(files,
                                                                   itertools.repeat(surveys),
                                                                   itertools.repeat(selectionParams),
                                                                   itertools.repeat(full_histogram),
                                                                   itertools.repeat(boxMaxDistance)
                                                               ))
    else:
        listOfSurveyContents = pool.starmap(surveyOneFile,zip(files,
                                                              itertools.repeat(surveys),
                                                              itertools.repeat(selectionParams),
                                                              itertools.repeat(full_histogram),
                                                              itertools.repeat(boxMaxDistance)
                                                          ))
    listOfSurveyContents=list(listOfSurveyContents)
    print("Generating surveys took {} seconds.".format(time.time()-start))
       
    #Format of listOfSurveyContents:
    #List of 1000 elements.
    #Each 'element' is a list of numSurveys elements, each element of which is a list of rows that belong to that
    #survey.

    #e.g. [
    #       [
    #         [rows for survey 1],
    #         [rows for survey 2],
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


def distsurvey(hugeFile,surveys,binsize,boxMaxDistance):
    galaxies = np.array(common.loadData(hugeFile,dataType = 'millPos'))
    masterhist = []
    firstrun = True
    #The EDGES were generated at the very top of this file. They are 27 coordinates - one for each box in a 3x3x3
    #cube. This lets me offset the millennium box into 27 boxes for periodic bounding conditions.

    """
    For example, when boxoffset = (0,0,0) you're in the center:
    . . .
    . X .
    . . .
    (each dot and the X represent a 500x500x500 Mpc/h cube, the entire Millennium simulation)
    
    and when boxoffset = (0,-1,0) you're down on the y axis:
    . . .
    . . .
    . X .

    and when it's (1,-1,0) you're here:
    . . .
    . . .
    . . X
    
    Of course I'm ignoring the 3rd coordinate Z here, but it does the same thing
    """
    
    for boxoffset in EDGES:
        distances = space.distance.cdist(galaxies + boxoffset * 500,surveys)
        numSurveys = distances.shape[1]
        assert(numSurveys == len(surveys))#This should never be a problem
        bins = genBins(binsize,boxMaxDistance)
        histogram = [] 
        for i in range(numSurveys):
            thisSurveyDistances = distances[:,i]
            #This means take distances for all the galaxies, but only the ith survey
            #That :, works since the first dimension of distances is the galaxies and the second dimension is surveys
            hist,edges = np.histogram(thisSurveyDistances, bins)
            histogram.append(hist)
            #So here we're counting the number of galaxies in this box in this bin of the survey. Later (back in
            #selectrun) we'll add together all the boxes to get a complete histogram.
        if firstrun:
            masterhist = histogram
        else:
            for i in range(len(masterhist)):
                masterhist[i] = histogram[i]+masterhist[i]
                #Add the histograms from each superbox together.
        firstrun = False
    #Notify the user that we're making progress
    print(".",end="",flush=True)
    return np.array(masterhist)

#@profile
def surveyOneFile(hugeFile,surveys,selectionParams,histogram,boxMaxDistance):
    """
    Given the original data, distances, wanted numbers, and other parameters we actually generate the
    mock surveys. This is currently the biggest bottleneck of the program, but I'm not entirely sure why.
                  ^ That comment might be outdated (It's not). Since I wrote it, I've rearranged the code a lot for
                  improved efficiency, including using the GPU for some otherwise expensive calculations.
    """
    #Set up variables
    rng = np.random.RandomState() #Make a process-safe random number generator
    first = True
    mastersurvey = []
    galax_pos =  np.array(common.loadData(hugeFile,dataType = 'millPos'))
    #Keep in mind that hugefile isn't really a huge file. It's a subbox of the millennium simulation, 50x50x50 mpc
    for boxoffset in EDGES:
        distances = space.distance.cdist(galax_pos + boxoffset * 500,surveys) #Recomputing the distance is faster
        #than saving it and reusing it. Weird, right?
        
        surveyContent = [[] for i in range(distances.shape[1])] #Make the skeleton structure for the end result
        galaxies = common.loadRawMillHybrid(hugeFile,boxoffset) #Load the galaxies !! 13.9% !!
        #WARNING: ^ Function assumes 500x500x500 box
        binsize = selectionParams['info']['shell_thickness'] #Load the size of a bin
        bins = genBins(binsize,boxMaxDistance)
        numSurveys = distances.shape[1]
    
        #Do calculations
        tdistBin =(np.digitize(distances.flatten(),bins)-1).reshape(distances.shape)
        wantNum = compute_want_density( ((tdistBin+1)*binsize)-binsize/2  ,binsize,**(selectionParams["constants"]))
        #FYI: Not really computing a density anymore. Haven't changed the name.
    
        #!!!!IMPORTANT!!!!!
        #tdistBin and distBin are the indexes of the bin each galaxy goes into. So if you're using bin 10 and
        #are looking at a galaxy with distance 38, it's in the 3rd bin (0 = 0-10, 1= 10-20, 2=20-30, 3=30-40)
        
        #Remember these are only the galaxies in the subbox we're looking at (surveyONEFILE)
        
        distBin = np.transpose(tdistBin)# !! 6 % !!
        histogram = np.concatenate((histogram,np.zeros((100,1))-1),axis=1)
        #When a galaxy is outside chop, its "number of galaxies around here" is set to -1
        #That makes the density negative
        #Which makes the probability negative so it is impossible for that galaxy to be selected.

        something = [histogram[n][distBin[n]] for n in range(numSurveys)]
        #This is the number of galaxies in the same bin as the galaxy and in the same box as the galaxy
        originalCount = np.transpose(np.array(something)) # 2.2%
        
        probability = wantNum / originalCount
        #Wantnum and originalcount are numpy arrays that each have one number for each galaxy in the subbox, for
        #each survey. So we get one probability per galaxy. Although really we only have one probability per bin
        #because of the way I've passed information to the compute_want_"density" function
        dice = rng.random_sample(probability.shape)
        #random numbers between 0 and 1
        toAdd = dice < probability #Boolean array - true for galaxies that are selected
        
        arrGalaxies = np.array(galaxies,dtype="object") #dtype object because we want to store the millennium text
        #but we want to have np fancy indexing.
        surveyContent = [arrGalaxies[toAdd[:,n]] for n in range(numSurveys)]
        print(".",end="",flush=True)
        if first:
            mastersurvey = surveyContent
            first = False
        else:
            for i in range(len(surveyContent)):
                mastersurvey[i] = np.concatenate((mastersurvey[i],surveyContent[i]))
    print("!")
    return mastersurvey



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
    #WAAAY more overhead, but as far as I know, also way faster in the long run
    #positionList = millennium.getAllPositions()
    for i in range(numSurveys):
        while True:
            #Note: this is basically the equivalent of bogosort as far as algorithm efficiency
            #is concerned. If you try to cram too many surveys into a box it will fail. There's a built in
            #failsafe that detects infinite looping by failing after it tries too many times. (currently 10000)
            #galaxyCoord = random.choice(positionList)
            galaxy=millennium.getARandomGalaxy()
            galaxyCoord = (galaxy.x,galaxy.y,galaxy.z)
            #distances = [math.sqrt((r1[0]-r2[0])**2+
            #                       (r1[1]-r2[1])**2+
            #                       (r1[2]-r2[2])**2)
            #                 for r1,r2 in zip(itertools.repeat(galaxyCoord),surveys)]
            if len(surveys) != 0:
                distances = space.distance.cdist([galaxyCoord],surveys)
            else:
                distances = np.array([])
            
            if np.all(distances > separation):
                #All is officially the 'coolest function ever.' All of an empty list is true!
                surveys.append(galaxyCoord)
                break
            else:
                numCatches += 1
            if numCatches % 500 == 0:
                print("So far we have tried {} times, and have {} surveys".format(numCatches,len(surveys)))
            #if numCatches > 100000:
            #    raise RuntimeError("We're probably in an infinite loop. Try reducing the number of surveys to make.")
    print("Caught {}!".format(numCatches))
    print([list(survey) for survey in surveys])
    return surveys

if __name__ == "__main__":
    arrrghs = common.parseCmdArgs([['settings'],['-g','--gpu']
                               ],
                                  ['Settings json file','use pyCUDA when GPU is available'
                                  ],
                                   [str,'bool'])

    selectrun(arrrghs)
