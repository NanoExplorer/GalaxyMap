"""
This file contains common functions used by my millenium programs. It contains mostly housekeeping functions
like loading files and outputting data.

Function reference:
    gensettings(args)
        takes an args paramater from argparse (which is never used) and writes a sample settings json file
        to settings.json in the current folder.
    getsettings(filename)
        Gets json settings stored in the filename, parses them and returns them as a python dictionary
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

def gensettings(args):
    if(input("Are you sure you want to overwrite settings.json? (y/n)")=='y'):
        with open("settings.json",'w') as settings:
            settingsDict = {'filename':'BoxOfGalaxies.csv','min_x':1.0,'max_x':10.0,'step_size':0.05,'num_runs':8}
            settings.write(json.dumps(settingsDict,
                                   sort_keys=True,
                                   indent=4, separators=(',', ': ')))
    exit()

def getsettings(filename):
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


def loadCSVData(filename):
    """
    Loads galaxy data from a CSV file. Assumes that the data is in the same format that my csv box was in,
    that is that X, Y, and Z coordinates are in rows 14,15, and 16 respectively.
    """
    print("Loading Coordinates...")

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

def writecsv(xslist,yslist):
    assert(len(xslist)==len(yslist))
    with open("./out2.csv",'w') as csv:
        for row in range(len(xslist[0])):
            line = ""
            for cell in range(len(xslist)):
                line = line + str(xslist[cell][row]) + ',' + str(yslist[cell][row])+ ','
            line = line + '\n'
            csv.write(line)
