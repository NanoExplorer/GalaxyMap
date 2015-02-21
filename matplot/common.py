"""
This file contains common functions used by my millenium programs. It contains mostly housekeeping functions
like loading files and outputting data.

Function reference:
    getdict(filename)
        Gets json settings stored in the filename, parses them and returns them as a python dictionary
    gensettings(args)
        takes an args paramater from argparse (assumed to have a  and writes a sample settings json file
        to settings.json in the current folder.
    loadCSVData(filename)
        Loads galaxy coordinates from a CSV file.
        Returns (xs, ys, zs) as a tuple
    makeplot(xs,ys,title,xl,yl)
        makes a matplotlib plot with the given X values, Y values, title, x label and y label.
    writecsv(xslist,yslist)
        VERY ESOTERIC USE WITH EXTREME CAUTION.
        Kept for legacy reasons.
        Writes a csv file with alternating X and Y columns, where x is a list of lists of x coordinates
        and yslist is a list of lists of y coordinates.

"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.backend_pdf as pdfback
import numpy as np
import json
import matplotlib.pyplot as plt
import os.path

def gensettings(args):
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

def getdict(filename):
    jsondict = None
    with open(filename,'r') as settings:
        jsondict = json.loads(settings.read())
        #reads the entire settings file with .read(), then loads it as a json dictionary, and stores it into jsondict
    return jsondict



def makeplot(xs,ys,title,xl,yl):
    fig = plt.figure(figsize=(4,3),dpi=100)
    ax = fig.add_subplot(111)
    ax.loglog(xs,ys,'o')
    ax.set_title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

def makeplotWithErrors(data,title,xl,yl):
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
    return name + '_{}_{}_{}.box'.format(xi,yi,zi)

def loadCSVData(filename):
    """
    Loads galaxy data from a CSV file. Assumes that the data is in the same format that my csv box was in,
    that is that X, Y, and Zelp='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                                           const=sum, default=max,
                                           help='sum the integers (default: find the max)')

    args = parser.parse_coordinates are in rows 14,15, and 16 respectively.
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

def loadCSVData(filename):
    """
    Loads galaxy data from a CSV file. Assumes that the data is in the same format that my csv box was in,
    that is that X, Y, and Zelp='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                                           const=sum, default=max,
                                           help='sum the integers (default: find the max)')

    args = parser.parse_coordinates are in rows 14,15, and 16 respectively.
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

def writecsv(xslist,yslist):
    assert(len(xslist)==len(yslist))
    with open("./out2.csv",'w') as csv:
        for row in range(len(xslist[0])):
            line = ""
            for cell in range(len(xslist)):
                line = line + str(xslist[cell][row]) + ',' + str(yslist[cell][row])+ ','
            line = line + '\n'
            csv.write(line)
