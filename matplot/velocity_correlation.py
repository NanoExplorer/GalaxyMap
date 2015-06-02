import common
import scipy.spatial.cKDTree as cKDTree
import numpy as np
import math

def psiOneNumerator(rv1, rv2, dTheta):
    """
    Calculates \psi 1's numerator as defined in the Gorski paper
    This is one iteration. So call this function a billion times then add up all the results.
    This function could be a good candidate for GPU-ing
    """
    return rv1*rv2*math.cos(dTheta)

def psiOneDenominator(dTheta):
    return (math.cos(dTheta))**2

def psiTwoNumerator(rv1,rv2,theta1,theta2):
    return rv1*rv2*math.cos(theta1)*math.cos(theta2)

def psiTwoDenominator(theta1,theta2,dTheta):
    return math.cos(theta1)*math.cos(theta2)*math.cos(dTheta)

def main(args):
    """
    Grab the CF2 file, chug it into cartesian (automatically done in common.py now!), plug into cKDTree, grab pairs
    plug information about pairs into psi functions, sum them, return values.
    """

if __name__ == "__main__":
    print("Doesn't run standalone - use galaxy.py instead")
