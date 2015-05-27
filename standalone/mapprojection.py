import matplotlib

matplotlib.use("TkAgg")

import matplotlib.backends.backend_pdf as pdfback
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import argparse




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", help="The input CF2 whitespace-delimited file",type=str)
    parser.add_argument("-o","--output", help="Write the resulting plot to a PDF file instead of displaying interactively",type=str)
    args = parser.parse_args()
    
    mag = 1 #could be a list. Dictates the size of each plotted point

    galaxies = loadCF2Data(args.inputFile)

    #Convert latitude and longitude (degrees) into radians, and move them to the ranges used by the plot
    #For longitude, we want it in the range -pi to pi instead of 0 to 2pi
    thetas = [galaxy.lon*(pi/180)-pi for galaxy in galaxies]
    phis = [galaxy.lat*(pi/180) for galaxy in galaxies]

    print("Generating plots...")
    fig=plt.figure(figsize=(8,4.5), dpi=180)
    ax = fig.add_subplot(111, projection='hammer')

    #Plot the data
    ax.scatter(thetas, phis, s=mag, color = 'r', marker = '.', linewidth = "1")

    #Set titles and axis labels
    ax.set_title('Angular Distribution of Galaxies',y=1.08)
    plt.xlabel('Galactic Longitude')
    plt.ylabel('Galactic Latitude')

    #Show the grid
    plt.grid(True)
    
    
    if args.output:
        print("Saving plots...")
        with pdfback.PdfPages(args.output) as pdf:    
            pdf.savefig(fig)
    else:
        plt.show()
    print("Done!")

def loadCF2Data(filename):
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


if __name__ == "__main__":
    main()
