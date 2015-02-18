import common
import time
import numpy as np
import scipy.spatial as space
import math
from multiprocessing import Pool
#import random
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
def hamest(unique,min_value,max_value,step,actual_list,box_info):
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

    Also, that above paragraph is wrong, and we'll have to do something to fix it in the future. It will be good
    to be able to have overlapping shells.
    """
    #We have to make the KD tree here because if we want to run an instance of this function on each processor
    #we need to make sure that we aren't passing any cKD trees into it. (Pickle drives the multiprocessing
    #argument passing, and a cpython object cannot be handled by pickle)
    cubic_min, cubic_max, num_galax = box_info
    rng = np.random.RandomState()
    random_list = rng.uniform(cubic_min,cubic_max,(num_galax,3))        
    #    with open(str(unique),'w') as dmp:
    #        dmp.write(str(random_list))

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
    print('.',end="",flush=True)
    return list(zip(xs,dxs,correlations))

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
        

def mainrun(args):
    print("Setting things up...")
    master_bins = []
    master_corrs = []
    settings = common.getdict(args.settings)
    filename = settings["filename"]
    min_x =    settings["min_x"]
    max_x =    settings["max_x"]
    step_size =settings["step_size"]
    runs = settings["num_runs"]
    print("Extracting galaxies...")
    xs, ys, zs = common.loadCSVData(filename)
    cubic_min = min(min(xs),min(ys),min(zs))
    cubic_max = max(max(xs),max(ys),max(zs))
    num_galax = len(xs)
    assert(len(xs) == len(ys) == len(zs))
    actual_galaxies = np.array(list(zip(xs,ys,zs)))
    print("Computing correlation function...")
    results = []
    start = time.time()
    print("    ",end="",flush=True)
    unique = range(runs)
    pool=Pool(processes=NUM_PROCESSORS)
    correlation_func_of_r = list(pool.starmap(hamest,list(zip(unique,
                                                              itertools.repeat(min_x),
                                                              itertools.repeat(max_x),
                                                              itertools.repeat(step_size),
                                                              itertools.repeat(actual_galaxies),
                                                              itertools.repeat((cubic_min,cubic_max,num_galax))))))
                                                              #This tuple here  ^         ^         ^
                                                              #exists to pass the information to build the random
                                                              #data set to the function.
                                                         
    """
    the structure of correlation_func_of_r is confusing right now, so I'll write it out.
    [
      [ (x,dx,y), (x,dx,y), ... (x,dx,y) ], <- run 0
      [ (x,dx,y), (x,dx,y), ... (x,dx,y) ], <- run 1
      [ (x,dx,y), (x,dx,y), ... (x,dx,y) ]  <- run 2
    ]
    
    """
                                                              
    print("That took {:.2f} seconds".format(time.time()-start))
    print("Computing statistics...")
    """
    OK, so Correlation Func of r is a list containing tuples of xs and ys (and dxs).
    To compute the error bars, we'll want to take the first y values out of all the tuples, find the standard
    deviation, multiply by two to calculate the error bars for each number, then find the standard deviation.
    Use that informaton to build a graph!
    """
    final_data = []
    for x_value in range(len(correlation_func_of_r[0])): 
        ys_for_this_x = [] #a list of the y values of a specific x value
        for y in range(len(correlation_func_of_r)):
            ys_for_this_x.append(correlation_func_of_r[y][x_value][2])
        final_data.append((correlation_func_of_r[0][x_value][0],
                           correlation_func_of_r[0][x_value][1],
                           np.average(ys_for_this_x),
                           np.std(ys_for_this_x)))
    

    print("Complete.")
    common.makeplotWithErrors(final_data,"Correlation function of distance r","Distance(Mpc/h)","correlation")
                     
if __name__ == "__main__":
    print("This python file does not run as a script. Instead use:")
    print("python galaxy.py correlation settings.json")
    print("where settings.json is your settings file.")
