print("Loading...")
import common
import time
import numpy as np
import scipy.spatial as space
import math
#from multiprocessing import Pool
#import random
import argparse
import itertools
NUM_PROCESSORS = 8

#from mayavi import mlab

"""
def d_p_est(r,dr,actual_kd,random_kd):
    #from http://ned.ipac.caltech.edu/level5/March04/Jones/Jones5_2.html
    #Nrd = N so the factor Nrd/N = 1 and will be left out.
    #DD(r) = average number of pairs
    #DR(r) = average num pairs between random and actual
    lower = r-(dr/2)
    assert(lower >= 0)
    upper = r+(dr/2)

    DD = actual_kd.count_neighbors(actual_kd,np.array([lower,upper]))
    DR = actual_kd.count_neighbors(random_kd,np.array([lower,upper]))
    print('.',end="",flush=True)
    return ((DD[1]-DD[0])/(DR[1]-DR[0]))-1
"""
def hamest(min_value,max_value,step,actual_list,random_list):
    """
    Notes: step = 2*dx
    min_value - dx >> 0 (or else we find a distance that is slightly greater than zero
    and we end up with a zero galaxy count and a divide by zero error)
    
    if min value is 1 and max value is 6 and step is one, we should generate a range
    [1,2,3,4,5,6(!!!!)]
    then subtract step/2 for
    [.5,1.5,2.5,3.5,4.5,5.5]
    so that the range generated ends up being in the middle of each pair (list[0],list[1]) or (list[1],list[2]) etc
    of values in the resultant list.
    """
    #We have to make the KD tree here because if we want to run an instance of this function on each processor
    #we need to make sure that we aren't passing any cKD trees into it. (Pickle drives the multiprocessing
    #argument passing, and a cpython object cannot be handled by pickle)
    actual_kd = space.cKDTree(actual_list,3)
    random_kd = space.cKDTree(random_list,3)
    num_elements = int((max_value-min_value)/step)+1+1 #one because range is not inclusive,
                                                       #one because int rounds down (and we want an EXTRA element)
    intervals = [((x*step)+min_value)-(step/2) for x in range(num_elements)]
    #This one needs the extra value at the end for producing intervals from first-dx to last+dx
    
    xs = [(x*step)+min_value for x in range(num_elements-1)]
    #Here we DON'T want that extra element, so we       ^ subtract one from num_elements
    #this will be the desired list of x values
    
    check_list = np.array(intervals)
    lower = min(check_list)
    assert(lower >= 0)
    DDs = actual_kd.count_neighbors(actual_kd,check_list)
    DRs = actual_kd.count_neighbors(random_kd,check_list)
    RRs = random_kd.count_neighbors(random_kd,check_list)
    dxs = itertools.repeat(step/2) #error in each x value
    correlations = calculate_correlations(DDs,DRs,RRs)
    
    #return value looks like this: a list of tuples, (x value, x uncertainty, y value)
    return zip(xs,dxs,correlations)

def calculate_correlations(DDs,DRs,RRs):
    results = []
    for index in range(len(DDs)-1):
        DDr = DDs[index+1]-DDs[index]
        #DDr, DRr, and RRr are all "number of objects in a shell" not "number of
        #objects closer than". This function converts from "closer than" to "in a shell"
        DRr = DRs[index+1]-DRs[index]
        RRr = RRs[index+1]-RRs[index]
        results.append((DDr*RRr)/(DRr**2)-1)
        #This is the formula for a hamilton estimator from http://ned.ipac.caltech.edu/level5/March04/Jones/Jones5_2.html
    return results

    

def unwrap(zvals):
    xs = []
    ys = []
    for tup in zvals:
        xs.append(tup[0])
        ys.append(tup[2])
    return (xs,ys)
        
def main():
    #Handle command line arguments
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_gensettings = subparsers.add_parser('sample_settings', help="generates a sample settings file")
    parser_gensettings.set_defaults(func=common.gensettings)
    parser_run = subparsers.add_parser('run', help="runs the correlation function")
    parser_run.add_argument("settings",help="Read in settings from this file.", type=str)
    parser_run.set_defaults(func=mainrun)
    
    args = parser.parse_args()

    function = None
    try:
        function =args.func
    except AttributeError:
        parser.print_help()
        exit()
    function(args)

def mainrun(args):
    master_bins = []
    master_corrs = []
    settings = common.getsettings(args.settings)
    filename = settings["filename"]
    min_x =    settings["min_x"]
    max_x =    settings["max_x"]
    step_size =settings["step_size"]
    xs, ys, zs = common.loadCSVData(filename)
    cubic_min = min(min(xs),min(ys),min(zs))
    cubic_max = max(max(xs),max(ys),max(zs))
    num_galax = len(xs)
    assert(len(xs) == len(ys) == len(zs))
    actual_galaxies = np.array(list(zip(xs,ys,zs)))
    print("    Generating random data set...")
    random_galaxies = np.random.uniform(cubic_min,cubic_max,(num_galax,3))
    print("    Computing correlation function...")
    start = time.time()
    correlation_func_of_r = hamest(min_x,max_x,step_size,actual_galaxies,random_galaxies)
    print("That took {:.2f} seconds".format(time.time()-start))
    print("Complete.")
    xs,ys = unwrap(correlation_func_of_r)
    common.makeplot(xs,ys,"Correlation function of distance r","Distance(Mpc/h)","correlation")
                     
if __name__ == "__main__":
    main()
