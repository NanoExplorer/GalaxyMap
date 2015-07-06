import common
import time
import numpy as np
import scipy.spatial as space
import math
from multiprocessing import Pool
#import random
import itertools
NUM_PROCESSORS = 8
MAXIMUM_ERROR = .1
MINIMUM_SAMPLE = 1/(MAXIMUM_ERROR**2)

np.seterr(divide='ignore',invalid='ignore')
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
def correlate_box(boxinfo, intervals):
    #load in the subbox
    xs, ys, zs = common.loadData(boxinfo[0])
    #grab its THEORETICAL minimum and maximum values:
    #Need to add detection of actual minimum and maximum values.
    #I might already have the logic in the box cutter function...
    rect_min = boxinfo[1][0]
    rect_max = boxinfo[1][1]
    #and its length
    num_galax = len(xs)
    #make sure we don't have a jagged array somehow
    #assert(num_galax == len(ys) == len(zs))
    #Never had a problem with that
    actual_galaxies = np.array(list(zip(xs,ys,zs)))
    #Make a thread-safe random number generator
    rng = np.random.RandomState()
    #and use it to make a list of random galaxy positions
    random_list = np.array(list(zip(rng.uniform(rect_min[0],rect_max[0],num_galax),
                                    rng.uniform(rect_min[1],rect_max[1],num_galax),
                                    rng.uniform(rect_min[2],rect_max[2],num_galax))))
    #make kd trees        
    actual_kd = space.cKDTree(actual_galaxies,3)
    random_kd = space.cKDTree(random_list,3)
    DDs = actual_kd.count_neighbors(actual_kd,intervals)
    print("This is WRONG - each pair is duplicated, and pairs are created between a galaxy and itself. So, to get the actual number of pairs, take the count_neighbors, subtract len(actual_galaxies), and divide by two.")
    exit()
    DRs = actual_kd.count_neighbors(random_kd,intervals)
    RRs = random_kd.count_neighbors(random_kd,intervals)
    #RDs = random_kd.count_neighbors(actual_kd,intervals)
    #Turns out that RDs == DRs always
    #Just think about it.
    print('.',end='',flush=True)
    return((DDs,DRs,RRs))
    

def makeSubDataPerBox(DDs, DRs, RRs, xs, ldrs, rdrs):
    localxs = list(xs)
    localLdrs = list(ldrs)
    localRdrs = list(rdrs)
    hamilton = hamest(DDs,DRs,RRs)
    davis_peebles = dpest(DDs,DRs,RRs)
    landy_szalay = lsest(DDs,DRs,RRs)
    random_point = randomPointCorrelation(DDs,DRs,RRs)
    if hamilton is None or davis_peebles is None or landy_szalay is None:
        #print("Zero correlation encountered!")
        return None
    #checkForBadNums(hamilton, davis_peebles, landy_szalay, localxs, localLdrs,localRdrs,random_point)
    return {"rs":localxs,
            "dr_left":localLdrs,
            "dr_right":localRdrs,
            "Hamilton":hamilton,
            "Davis_Peebles":davis_peebles,
            "Landy_Szalay":landy_szalay,
            "Random_Correlation":random_point,
            "DDs":[int(num) for num in DDs],
            "DRs":[int(num) for num in DRs],
            "RRs":[int(num) for num in RRs]
    }
#deprecated in favor of checkSampleSize()
# def checkForBadNums(*args):
#     #The lists always need to have the same length!
#     for i in range(len(args[0])-1,-1,-1):
#         #go through the lists from the back to the front.
#         bad = False
#         for arg in args:
#             if not math.isfinite(arg[i]):
#                 bad = True
#                 break
#         if bad:
#             for arg in args:
#                 del arg[i]
            
def calculate_correlations(args):
    """
    args = (unique, boxinfo, numpoints, dr, step size, minimum radius, type of step to use for xs)
    
    min_value - dr >> 0 (or else we find a distance that is slightly greater than zero
    and we end up with a zero galaxy count and a divide by zero error)

    Actually, I've gotten to the point where divide by zero errors are handled by stats.py
    so don't worry about it.
    """

    unique, boxinfo, numpoints, dr, step_size, min_r, step_type = args
    
    if step_type == "lin":
        inter_fun = common.lin_intervals
    elif step_type == "log":
        inter_fun = common.log_intervals
    else:
        raise ValueError("step_type {} undefined".format(step_type))
    xs,intervals = inter_fun(min_r, step_size, numpoints, dr)
    left_drs = getLeftdrs(xs,intervals)
    right_drs = getRightdrs(xs,intervals)
    #Left and right drs are used when plotting error bars
    check_list = np.array(intervals)
    lower = min(check_list)
    #assert(lower >= 0)

    DDs = [0 for x in range(len(check_list))]
    DRs = [0 for x in range(len(check_list))]#These arrays should contain total numbers of pairs.
    RRs = [0 for x in range(len(check_list))]
    

    #make a list of boxes:
    boxes = list(boxinfo['list_of_files'].items())
    pool = Pool(processes = NUM_PROCESSORS)
    list_of_DDsDRsRRs = pool.starmap(correlate_box,zip(boxes, itertools.repeat(check_list)))
        #The box that we're passing them isn't actually *just* a box. Instead it's a tuple
        #(filename, [[minx,miny,minz],[maxx,maxy,maxz]])
        #the tuple contains the filename of the box, the coordinates of the box's minimum corner
        #and the coordinates of the box's maximum corner.
    dataPerBox = {}
    num_bad_points = 0
    for x in range(len(list_of_DDsDRsRRs)):
        item = list_of_DDsDRsRRs[x]
        boxfile = boxes[x][0]


        #Make a local copy of these things for use in culling.

        miniDDs, miniDRs, miniRRs = item

        assert(len(DDs) == len(DRs) == len(RRs) == len(miniDDs) == len(miniDRs) == len(miniRRs))
        #I think I'm just paranoid.
        
        #Make DDs, DRs, and RRs contain the *total* number of pairs found.
        for i in range(len(miniDDs)):
            DDs[i] = DDs[i] + miniDDs[i]
            DRs[i] = DRs[i] + miniDRs[i]
            RRs[i] = RRs[i] + miniRRs[i]
            
        
        data = makeSubDataPerBox(miniDDs, miniDRs, miniRRs, xs, left_drs, right_drs)
        if data is not None:
            dataPerBox[boxfile] = data
        else:
            num_bad_points += 1
    print("Number of bad runs: {}".format(num_bad_points))

    correlations = makeSubDataPerBox(DDs, DRs, RRs, xs, left_drs, right_drs)
    dataPerBox['ALL_BOXES'] = correlations
    #return value looks like this: a list of tuples, (x value, x uncertainty, y value)
    print('Run {} complete.'.format(unique))
    return dataPerBox

def getLeftdrs(xs,intervals):
    return [xs[i] - intervals[i*2] for i in range(len(xs))]

def getRightdrs(xs,intervals):
    return [intervals[i*2+1] - xs[i] for i in range(len(xs))]

def checkSampleSize(Dd,Dr,Rr,name):
    """
    if Dd < MINIMUM_SAMPLE:
        print("Warning: Small sample size encountered in {} correlation calculation, {} value = {}.".format(name,
                                                                                                            "DD",
                                                                                                            Dd))
        print("Consider increasing bin size.")
    if Dr < MINIMUM_SAMPLE:
        print("Warning: Small sample size encountered in {} correlation calculation, {} value = {}.".format(name,
                                                                                                            "DR",
                                                                                                            Dr))
        print("Consider increasing bin size.")
    if Rr < MINIMUM_SAMPLE:
        print("Warning: Small sample size encountered in {} correlation calculation, {} value = {}.".format(name,
                                                                                                            "RR",
                                                                                                            Rr))
        print("Consider increasing bin size.")"""
    if Dd == 0 or Dr == 0 or Rr == 0:
        return True
    else:
        return False

def hamest(DDs,DRs,RRs):
    results = []
    for index in range(0,len(DDs),2):
        DDr = DDs[index+1]-DDs[index]
        #DDr, DRr, and RRr are all "number of objects in a shell" not "number of
        #objects closer than". This function converts from "closer than" to "in a shell"
        DRr = DRs[index+1]-DRs[index]
        RRr = RRs[index+1]-RRs[index]
        if checkSampleSize(DDr,DRr,RRr,"hamilton"):
            return None
        results.append((DDr*RRr)/(DRr**2)-1)
        #This is the formula for a hamilton estimator from http://ned.ipac.caltech.edu/level5/March04/Jones/Jones5_2.html
    return results
    
def dpest(DDs,DRs,RRs):
    results = []
    for index in range(0,len(DDs),2):
        DDr = DDs[index+1]-DDs[index]
        #DDr, DRr, and RRr are all "number of objects in a shell" not "number of
        #objects closer than". This function converts from "closer than" to "in a shell"
        DRr = DRs[index+1]-DRs[index]
        RRr = RRs[index+1]-RRs[index]
        if checkSampleSize(DDr,DRr,RRr,"Davis Peebles"):
            return None
        results.append((DDr/DRr)-1)
        #This is the formula for a Davis and Peebles estimator from http://ned.ipac.caltech.edu/level5/March04/Jones/Jones5_2.html
        #Nrd = N
    return results


def lsest(DDs,DRs,RRs):
    results = []
    for index in range(0,len(DDs),2):
        DDr = DDs[index+1]-DDs[index]
        #DDr, DRr, and RRr are all "number of objects in a shell" not "number of
        #objects closer than". This function converts from "closer than" to "in a shell"
        DRr = DRs[index+1]-DRs[index]
        RRr = RRs[index+1]-RRs[index]
        if checkSampleSize(DDr,DRr,RRr,"Landy Szalay"):
            return None
        results.append(1+(DDr/RRr)-2*(DRr/RRr))
        #This is the formula for a Landy and Szalay estimator from http://ned.ipac.caltech.edu/level5/March04/Jones/Jones5_2.html
        #Nrd = N
        #number of randomly uniform points = number of 'actual' points
    return results

def randomPointCorrelation(DDs,DRs,RRs):
    results = []
    for index in range(0,len(DDs),2):
        DDr = DDs[index+1]-DDs[index]
        #DDr, DRr, and RRr are all "number of objects in a shell" not "number of
        #objects closer than". This function converts from "closer than" to "in a shell"
        DRr = DRs[index+1]-DRs[index]
        RRr = RRs[index+1]-RRs[index]
        if RRr == 0:
            results.append(0)
        else:
            results.append((DRr/RRr)-1)
        #This takes the number of galaxies in a shell around a random point
        #and compares it to the number of random points in a shell around the random point.
        #Nrd = N
        #number of randomly uniform points = number of 'actual' points
    return results


    
"""
def unwrap(zvals):
    xs = []
    ys = []
    for tup in zvals:
        xs.append(tup[0])
        ys.append(tup[2])
    return (xs,ys)
   """     

def mainrun(args):
    print("Setting things up...")
    all_settings = common.getdict(args.settings)

    boxname =   all_settings["boxname"]
    settings = all_settings["Correlation"]
    numpoints = settings["numpoints"]
    dr =        settings["dr"]
    runs =      settings["num_runs"]
    min_r =     settings["min_r"]
    step_size = settings["step_size"]
    step_type = "lin"
    try:
        step_type = settings["step_type"]
    except KeyError:
        print("We've added a new argument \"step_type\" to the settings file.\nPlease update {} to include this argument.\nDefault value is \"lin\" for linear steps on the x axis.\nOther values are \"log\" for logarithmic point spacing. More types to come. Maybe.".format(args.settings))
    boxinfo = common.getdict(boxname)
    print("Computing correlation function...")
    argslist = [(x,
                 boxinfo,
                 numpoints,
                 dr,
                 step_size,
                 min_r,
                 step_type) for x in range(runs)]
    start = time.time()
    correlation_func_of_r = list(map(calculate_correlations,argslist))
    finish = time.time()-start
    print("That took {} seconds.".format(finish))

    """
    the structure of correlation_func_of_r:
    [
      [ (x1,dx1,y1), (x2,dx2,y2), ... (xn,dxn,yn) ], <- run 0
      [ (x1,dx1,y1), (x2,dx2,y2), ... (xn,dxn,yn) ], <- run 1
      [ (x1,dx1,y1), (x2,dx2,y2), ... (xn,dxn,yn) ]  <- run 2
    ]
    
                                                                  
    print("Computing statistics...")

    Correlation Func of r is a list containing tuples of xs and ys (and dxs).
    To compute the error bars, we'll want to take the y values out of all the tuples, grouped by x value,
    find the standard deviation, and multiply by two to calculate the error bars for each number.
    Then find the average of the y values to place the center point of the error bars.
    Finally, save that information to file and then use it to build a graph!

    We're actually going to move the "computing statistics" thingies to a different file.
    
    final_data = []
    for x_value in range(len(correlation_func_of_r[0])): 
        ys_for_this_x = [] #a list of the y values of a specific x value
        for y in range(len(correlation_func_of_r)):
            ys_for_this_x.append(correlation_func_of_r[y][x_value][2])
        final_data.append((correlation_func_of_r[0][x_value][0],
                           correlation_func_of_r[0][x_value][1],
                           np.average(ys_for_this_x),
                           2*np.std(ys_for_this_x)))
    """
    print("Complete.")
    dataFileName = settings['output_data_folder'] + boxname.split('/')[-1] + '---'
    dataFileName = dataFileName + args.settings.split('/')[-1].split('.')[0] + '---rawdata.json'
                            
                            
                            
    common.writedict(dataFileName,
                     {'raw_runs':correlation_func_of_r,
                      'settings':all_settings,
                      'time':finish})
                     
if __name__ == "__main__":
    print("This python file does not run as a script. Instead use:")
    print("python galaxy.py correlation settings.json")
    print("where settings.json is your settings file.")
    
