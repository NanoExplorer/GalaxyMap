"""
This file contains common functions used by my millenium programs. It contains mostly housekeeping functions
like loading files and outputting 
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.backend_pdf as pdfback
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import math
import scipy.spatial as space
import random
import argparse

def parseCmdArgs(argumentList,helpList,typeList):
    """Parses command-line arguments. Useful for moving away from the galaxy.py universal interface
    because it takes so much time to start up, and has to import everything. 
    """
    parser = argparse.ArgumentParser()
    for argument,theHelp,theType in zip(argumentList,helpList,typeList):
        if theType == 'bool':
            parser.add_argument(*argument,help=theHelp,action='store_true')
        else:
            parser.add_argument(*argument,help=theHelp,type=theType)
    return parser.parse_args()


def intervals(min_r,step_size,numpoints,dr,step_type):
    """
    returns xs, intervals where xs are the center points, intervals are the 'bin edges'
    In theory, this lets me make lots of mistakes, and it is currently preferred that you use
    the genBins function below instead.
    """
    if step_type == 'lin':
        return lin_intervals(min_r,step_size,numpoints,dr)
    elif step_type == 'log':
        return log_intervals(min_r,step_size,numpoints,dr)
    else:
        raise RuntimeError("Interval type {} not recognized".format(step_type))

def lin_intervals(min_r,step_size,numpoints,dr):
    """Generates linearly-spaced intervals, and their center points."""
    xs = [(min_r+step_size*x) for x in range(numpoints)]
    intervals = []
    for x in xs:
        intervals.append(x-(dr/2.0))
        intervals.append(x+(dr/2.0))
    return (xs, intervals)
    
def log_intervals(min_r,step_size,numpoints,dr):
    """Generates logarithmically spaced intervals and their center points
    dr is a measure of the size of each interval, as a percentage of the distance between the previous and
    next interval edges. if dr is .5 then exactly all of the range will be covered with zero overlap.
    """
    #The minus one exists so that when x is zero xs is equal to min_r
    xs = [(min_r + 10**(step_size*x) - 1) for x in range(numpoints)]
    intervals = []
    for i in range(numpoints):
        x = xs[i]
        intervals.append(x-(dr*((10**(step_size*i))*(1-10**(-step_size)))))
        intervals.append(x+(dr*((10**(step_size*i))*(10**(step_size)-1))))
    return (xs, intervals)

def genBins(min_r,numpoints,dr,step_type):
    """Generates bins that are compatible with sci- and num-py histogram functions.
    """
    if step_type == 'lin':
        return lin_bins(min_r,numpoints,dr)
    elif step_type == 'log':
        return log_bins(min_r,numpoints,dr)
    else:
        raise RuntimeError("Bin type {} not recognized".format(step_type))

def lin_bins(min_r,numpoints,dr):
    """Generates linearly-spaced bins that can be used with histogram functions."""
    xs = [min_r + dr*x for x in range(numpoints)]
    intervals = [(min_r-dr/2)+(dr*x) for x in range(numpoints + 1)]
    return (xs,intervals)
    
def log_bins(min_r,numpoints,dr):
    """Generates logarithmically spaced bins that can be used with histogram functions."""
    xs = [(min_r + 10**(dr*x) - 1) for x in range(numpoints + 1)]
    intervals = [x-(.5*((10**(dr*i))*(1-10**(-dr)))) for i,x in enumerate(xs)]
    return(xs,intervals)
    
def sphereVol(radius):
    """Returns the volume of sphere"""
    return (4/3)*(np.pi)*(radius**3)

def shellVolCenter(r, thickness):
    """Returns the volume of a shell centered on radius r, with specified thickness."""
    dr = thickness/2
    left = r-dr
    right = r+dr
    return shellVol(left,right)
        
def shellVol(r1,r2):
    """Returns the volume of a shell with inner edge r1 and outer edge r2"""
    return abs(sphereVol(r1)-sphereVol(r2))

def gensettings(args):
    """Makes a sample settings file called settings_<MODULENAME>.json
    The sample settings file contains the settings information from that module contained in the
    template_settings.json file.
    """
    module = args.module
    filename = "settings_{}.json".format(module)
    template = getdict("template_settings.json")
    if(not os.path.exists(filename) or input("Are you sure you want to overwrite {}? (y/n) ".format(filename))=='y'):
        #if the file doesn't exist, it goes ahead with out asking.
        #if the file does exist, then it asks.
        #Woo for boolean operator overloading!
        with open(filename,'w') as settings:
            settings.write(json.dumps(template[module],
                                      sort_keys=True,
                                      indent=4, separators=(',', ': ')))
    exit()

def writedict(filename, dictionary):
    """Given a filename and a dictionary (or list...), writes the dictionary to file as json.
    The dictionary can then be retrieved with the getdict function below.
    Setting pretty to false can make the file considerably smaller, at the cost of being almost unintelligable.
    """
    with open(filename,'w') as jsonfile:
        jsonfile.write(json.dumps(dictionary,
                                  sort_keys=True,
                                  indent=4, separators=(',', ': ')))
        


def getdict(filename):
    """Given the name of a json file, reads the file in and returns the object it contains
    """
    jsondict = None
    with open(filename,'r') as settings:
        jsondict = json.loads(settings.read())
        #reads the entire settings file with .read(), then loads it as a json dictionary, and stores it into jsondict
    return jsondict



def makeplot(xs,ys,title,xl,yl):
    """DEPRECATED. Avoid if possible."""
    fig = plt.figure(figsize=(4,3),dpi=100)
    ax = fig.add_subplot(111)
    ax.loglog(xs,ys,'o')
    ax.set_title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

def makeplotWithErrors(data,title,xl,yl):
    """DEPRECATED. Avoid if possible."""
    fig = plt.figure(figsize=(4,3),dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposx='clip')

    plt.errorbar([x[0] for x in data],[x[2] for x in data],xerr=[x[1] for x in data],yerr=[x[3] for x in data]) 
    ax.set_title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

def getBoxName(name, xi, yi, zi):
    """
    returns the name of the box given its stub name and the x, y, and z indices.
    Works on dicer-split box files
    """
    return name + '_{}_{}_{}.box'.format(xi,yi,zi)

def getBoxNameJackknife(name, ranint):
    """
    returns the name of the box given its stub name and random iteration.
    Works on jackknife-split box files.
    """
    return name + '_{}.box'.format(ranint)

def loadData(filename, dataType = "guess"):
    """
    Loads in galaxy data. Always returns a list of galaxies. Galaxies can be in multiple formats. Most of the
    time the format is merely a list [x,y,z]. However, in the case of CF2 data, the result is a list of CF2 galaxies
    with data type defined below. The miscFloat type returns a list of lists of floats.
    
    This method tries to guess what kind of data you're loading based on the extension on the filename.
    Override this using the dataType named argument.

    dataTypes:
        'dat': boxfiles that are like lasdamas box files
        'csv': boxfiles from millennium that have x, y, and z values in columns 14, 15, and 16
        'box': boxfiles created by dicer.py
        'miscFloat': csv files that contain only floats
        'CF2': read data from a CF2 or COMPOSITE survey
    """
    if dataType == "guess":
        dataType = filename.split('.')[-1].lower()
        if dataType == "txt":
            dataType = "CF2" #Note: This might not always be the case.
    if dataType == 'dat':
        return _loadDATData(filename)
    elif dataType == 'csv':
        return _loadCSVData(filename)
    elif dataType == 'box':
        return _loadBOXData(filename)
    elif dataType == 'miscFloat':
        return _loadCSVFloatData(filename)
    elif dataType == 'CF2':
        return _loadCF2Data(filename)
    elif dataType == 'millPos':
        return _loadMillenniumPositionalData(filename)
    elif dataType == 'millVel':
        return _loadMillenniumVelocityData(filename)
    elif dataType == 'millRaw':
        return _loadMillenniumRawLines(filename)

def _loadCSVFloatData(filename): 
    """
    Loads miscellaneous data from a csv file.
    Note: Assumes all data in the file are floats.
    """
    csvData = []
    with open(filename, "r") as boxfile:
        for line in boxfile:
            if line[0]!="#":#comment lines need to be ignored
                row = line.split(',')
                csvRow = []
                valid = True
                for cell in row:
                    try:
                        csvRow.append(float(cell))
                    except ValueError:
                        valid = False#sometimes the CSV file doesn't contain a number.
                                     #In that case, just skip that row.
                if valid:
                    csvData.append(csvRow)
    return csvData

def _loadMillenniumRawLines(filename): 
    """
    Loads the raw CSV lines from a CSV file
    """
    csvData = []
    with open(filename, "r") as boxfile:
        for line in boxfile:
            if line[0]!="#":#comment lines need to be ignored
                row = line.split(',')
                valid = True
                for cell in row[0:3]:
                    try:
                        x =float(cell)
                    except ValueError:
                        valid = False#sometimes the CSV file doesn't contain a number.
                                     #In that case, just skip that row.
                if valid:
                    csvData.append(line)
    return csvData

def _loadMillenniumVelocityData(filename):
    csvData = []
    with open(filename, "r") as boxfile:
        for line in boxfile:
            if line[0]!="#":#comment lines need to be ignored
                row = line.split(',')
                csvRow = []
                valid = True
                for cell in row[0:6]:
                    try:
                        csvRow.append(float(cell))
                    except ValueError:
                        valid = False#sometimes the CSV file doesn't contain a number.
                                     #In that case, just skip that row.
                if valid:
                    csvData.append(csvRow)
    return csvData
    
def _loadMillenniumPositionalData(filename): 
    """
    Loads the first three numbers from a csv file.
    With my millennium data, those three numbers are x,y, and z coordinates
    """
    csvData = []
    with open(filename, "r") as boxfile:
        for line in boxfile:
            if line[0]!="#":#comment lines need to be ignored
                row = line.split(',')
                csvRow = []
                valid = True
                for cell in row[0:3]:
                    try:
                        csvRow.append(float(cell))
                    except ValueError:
                        valid = False#sometimes the CSV file doesn't contain a number.
                                     #In that case, just skip that row.
                if valid:
                    csvData.append(csvRow)
    return csvData


def _loadBOXData(filename):
    """
    Loads galaxy data from a BOX file created by dicer.py
    """
#    print("Loading Coordinates...")

    xs = []#list of x coordinates of galaxies. The coordinates of galaxy zero are (xs[0],ys[0],zs[0])
    ys = []
    zs = []

    with open(filename, "r") as boxfile:
        for line in boxfile:
            row = line.split(',')
            xs.append(float(row[0]))
            ys.append(float(row[1]))
            zs.append(float(row[2]))
        return (xs,ys,zs)
def _loadCSVData(filename):
    """
    Loads galaxy data from a CSV file. Assumes that the data is in the same format that my csv box was in,
    that is that X, Y, and Z are in rows 14,15, and 16 respectively.
    """
#    print("Loading Coordinates...")

    xs = []#list of x coordinates of galaxies. The coordinates of galaxy zero are (xs[0],ys[0],zs[0])
    ys = []
    zs = []

    with open(filename, "r") as boxfile:
        for line in boxfile:
            if line[0]!="#":#comment lines need to be ignored
                try:
                    row = line.split(',')
                    xs.append(float(row[14]))
                    ys.append(float(row[15]))
                    zs.append(float(row[16]))
                except ValueError:
                    pass#sometimes the CSV file doesn't contain a number. In that case, just skip that row.
    return (xs,ys,zs)

def _loadDATData(filename):
    """
    Loads galaxy data from a DAT file. Assumes that the data is in the same format that my dat box was in,
    that is that X, Y, and Z are in columns 0, 1, and 2 respectively.
    """

    xs = []#list of x coordinates of galaxies. The coordinates of galaxy zero are (xs[0],ys[0],zs[0])
    ys = []
    zs = []

    with open(filename, "r") as boxfile:
        for line in boxfile:
            row = line.split()
            xs.append(float(row[0]))
            ys.append(float(row[1]))
            zs.append(float(row[2]))
    return (xs,ys,zs)

def _loadCF2Data(filename):
    """
    Loads galaxy data from a TXT file. Assumes that the data is in the same format
    as the CF2 and COMPOSITE surveys:
    two-space-delimited, with columns
            cz (km/s)
            distance (Mpc/h)
            radial velocity(km/s)
            error in radial velocity(km/s)
            Galactic Longitude (degrees)
            Galactic Latitude (degrees)
    """

    galaxies = [] 
    with open(filename, "r") as boxfile:
        for line in boxfile:
            row = line.split()
            floats = [float(x) for x in row]
            galaxies.append(CF2(floats))
    return galaxies

def writecsv(xslist,yslist):
    """Deprecated. I remember that this function is mostly useless, but I don't remember exactly what it does."""
    assert(len(xslist)==len(yslist))
    with open("./out2.csv",'w') as csv:
        for row in range(len(xslist[0])):
            line = ""
            for cell in range(len(xslist)):
                line = line + str(xslist[cell][row]) + ',' + str(yslist[cell][row])+ ','
            line = line + '\n'
            csv.write(line)
class CF2:
    """
    Data structure for holding a cf2 galaxy. The constructor takes a list of attributes,
    in the order given by the cf2 files. 
    """
    def __init__(self, data):
        self.cz = data[0]
        self.d = data[1]
        self.v = data[2]
        self.dv = data[3]
        self.lon = data[4]
        self.lat = data[5]
        self.doc = {"cz": "Redshift. Has units km/sec.",
                    "d": "Distance. Has units Mpc/h.",
                    "v": "Peculiar (radial) velocity in km/sec.",
                    "dv": "Error in peculiar velocity. Units: km/sec.",
                    "lon":"Galactic longitude in degrees.",
                    "lat":"Galactic latitude in degrees."}
        self.units = {"cz": "km/sec",
                    "d": "Mpc/h",
                    "v": "km/sec",
                    "dv": "km/sec",
                    "lon":"degrees",
                    "lat":"degrees"}
    
        self.theta = math.radians(self.lon-180)
        self.phi = math.radians(self.lat+90)
        self.x = self.d*math.sin(self.phi)*math.cos(self.theta)
        self.y = self.d*math.sin(self.phi)*math.sin(self.theta)
        self.z = self.d*math.cos(self.phi)
    def __str__(self):
        outstr = "CF2 galaxy at redshift {} km/s, distance {} Mpc/h,\n".format(self.cz,self.d)
        outstr += "with peculiar velocity {} km/s.\n".format(self.v)
        outstr += "This galaxy is in the sky at galactic latitude {} and longitude {} (deg)".format(self.lat,self.lon)
        return outstr
    def getRedshiftXYZ(self):
        redx = self.cz*math.sin(self.phi)*math.cos(self.theta)
        redy = self.cz*math.sin(self.phi)*math.sin(self.theta)
        redz = self.cz*math.cos(self.phi)
        return(redx,redy,redz)
        
class MillenniumFiles:
    """
    Handles certain operations with regards to boxes of millennium files. 
    """
    def __init__(self, boxLocation):
        """
        Takes in a directory, and initializes based on that directory and the files in it.
        The directory must have a file next to it called 'nameofdirectory_info.json' that contains
        box size information and box filename format information
        """
        #Make sure we're looking at a directory.
        boxIsDir = os.path.isdir(boxLocation)
        #self.files is always a list of files
        self.boxLocation = boxLocation
        #This class is designed for handling millennium directories, so throw an exception if it isn't a directory
        if boxIsDir:
            self.files = [boxLocation+fname for fname in os.listdir(boxLocation)]
        else:
            raise TypeError("The MillenniumFiles class may only be used on directories containing many sub-boxes.")

        #grab the informational json file. If it doesn't exist, throw the corresponding exception.
        boxInfoLoc = boxLocation.rstrip('/') + '_info.json'
        if os.path.isfile(boxInfoLoc):
            self.boxInfo = getdict(boxInfoLoc)
        else:
            raise RuntimeError("The box must have an associated informational JSON file located at {}!".format(boxInfoLoc))
        
        #I wish I could think of a better way of storing info files. For now,
        #every box I use this program on will have to just have an associated
        #json file like this.

        #Self.boxinfo contains three important pieces of information:
        #    the size of each box (a 3-element list/tuple) "box_size"
        #    the format string for finding a filename, using attributes xi, yi, and zi for indices "box_filename_format"
        #    the directory the box is sitting in, relative to this file (not quite as useful.)"box_directory"

    def getACloseGalaxy(self,r):
        """
        Returns a galaxy close to the (x,y,z) tuple 'r'
        Note: The galaxy returned is not guaranteed to be the closest galaxy to r, especially if r
        is close to a box edge. This method will be less computationally intensive than get "closest" galaxy
        """
        csvData = loadData(self.getBox(r), 'miscFloat')
        spatialInfo = []
        for row in csvData:
            galaxy = MillenniumGalaxy(row)
            spatialInfo.append((galaxy.x,galaxy.y,galaxy.z))
        kd_galaxies = space.cKDTree(spatialInfo)
        distance, index = kd_galaxies.query(r)
        return spatialInfo[index]

    def getARandomGalaxy(self):
        """Gets a random galaxy from the simulation. The randomness is not quite distributed properly -
        to fix, the getARandomBox function should choose weighted random based on the number of galaxies in each box.
        right now, galaxies in more sparsely populated boxes are slightly more likely to be chosen
        """
        box = self.getARandomBox()
        #Waterman's reservoir algorithm from Knuth's Art of Computer Programming and
        #http://stackoverflow.com/questions/3540288/how-do-i-read-a-random-line-from-one-file-in-python
        with open(box,'r') as f:
            line = next(f)
            while line[0] == 'x' or line[0] =='#':
                line = next(f)
            for num, templine in enumerate(f):
                if random.randrange(num + 2) == 0 and templine[0] != '#':
                    #Picks a random number from the range [0,num+2). if the number picked is zero,
                    #we replace the stored line with a new one.
                    line = templine
        return MillenniumGalaxy(line) # If this fails, that means a comment line probably slipped by somehow.
        
    def getARandomBox(self):
        """NOT A WEIGHTED RANDOM"""
        return random.choice(self.files)
            
    def getBox(self,r):
        """
        Returns the box (filename) that this point would be in
        """
        
        size = self.boxInfo["box_size"]
        xbox = int(r[0] / size[0])
        ybox = int(r[1] / size[1])
        zbox = int(r[2] / size[2])
        filename = self.boxInfo["box_filename_format"].format(xi = xbox, yi = ybox, zi = zbox)
        fullPath = self.boxLocation + filename
        assert fullPath in self.files #If this assertion fails, you're most likely outside the box.
        return fullPath

    def getAllPositions(self):
        """
        WARNING: This function may be subject to large amounts of memory usage and loading time.
        Use with caution and discretion.
        """
        positionList = []
        for box in self.files:
            posList = loadData(box,'millPos')
            positionList += posList
        return positionList

    def readABox(self, filename):
        return loadData(filename,'millPos')

    def getFiles(self):
        return self.files
        
        
class MillenniumGalaxy:
    def __init__(self,galaxy):
        """
        Data structure for holding galaxies from one type of millennium query.
        the parameter 'galaxy' is either a string (one line) directly from the millennium box
        or a list of attributes in the order presented by the millennium box
        """
        #If we get the entire csv line as a string, we can just split it and make sure everything's a float.
        if type(galaxy) is str:
            self.galaxList = [float(x) for x in galaxy.strip().split(',')[0:22]]
            #the 0:22 is there to prevent hanging commas from destroying everything
        elif type(galaxy) is list:
            #If we get a list, we assume it is already full of floats.
            self.galaxList = list(galaxy)
            #The list() function is used to make sure we have a copy of the galaxy list and not the original.
        else:
            raise TypeError("Millennium Galaxies can only be constructed with lists or csv strings.")
            
        #This next part is really ugly. I apologize.
        #If you think of a better way to do this, email me at christopher@rooneyworks.com
        #I'm thinking I could use a dictionary or something like that...
        #Maybe dynamically pull the field names from the comments at the beginning of the csv file.
        #The problem with that is it requires dynamic execution of arbitrary code, and I'm not quite
        #comfortable with that. I'm sure this will work just fine for now.
        #Or it requires using a dictionary, but then you have to reference everything as ["name"] instead of
        #just .name, and .name is SOOO much faster to type than ["name"]
        self.x = self.galaxList[0]
        self.y = self.galaxList[1]
        self.z = self.galaxList[2]
        self.velX = self.galaxList[3]
        self.velY = self.galaxList[4]
        self.velZ = self.galaxList[5]
        self.mvir = self.galaxList[6]
        self.rvir = self.galaxList[7]
        self.vvir = self.galaxList[8]
        self.vmax = self.galaxList[9]
        self.bulgeMass = self.galaxList[10]
        self.stellarMass = self.galaxList[11]
        self.mag_b = self.galaxList[12]
        self.mag_v = self.galaxList[13]
        self.mag_r = self.galaxList[14]
        self.mag_i = self.galaxList[15]
        self.mag_k = self.galaxList[16]
        self.mag_bBulge = self.galaxList[17]
        self.mag_vBulge = self.galaxList[18]
        self.mag_rBulge = self.galaxList[19]
        self.mag_iBulge = self.galaxList[20]
        self.mag_kBulge = self.galaxList[21]
    def toString(self):
        string = ""
        for data in self.galaxList:
            string = string + str(data) + ','
        string.strip(',')
        return string

        
def crossProductMatrix(r):
    r1 = r[0]
    r2 = r[1]
    r3 = r[2]
    return np.array([[0,-r3,r2],
                     [r3,0,-r1],
                     [-r2,r1,0]])
    #definition from http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
