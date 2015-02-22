#Takes in a file,
#chops it up into many files based on x y and z positions (boxes)
import common
import math
import json

def dice(args):
    #algorithm: go through the box and find its bounds
    #Divide that volume into sub-boxes using the x_divs, y_divs and z_divs from the settings file
    #go through the box again and test each data point and put it into the right boxfile(s)
    #remember that each box has a standard and an extended data set.
    #filename convention: "<NAME>_xindex_yindex_zindex(.extended).box"
    #settings convention: "<NAME>"

#Think of these as macro definitions. I want to be able to say sizes[Z]
#because I'll feel "cool" doing that.
    X = 0
    Y = 1
    Z = 2
    MIN= 0
    MAX = 1
    filebounds = dict()
    
    settings=common.getdict(args.settings)
    inFileName = settings["filename"]
    sizes = (settings["x_box_size"], settings["y_box_size"], settings["z_box_size"])
    radius = settings["expected_radius"]
    outFileName = settings["boxname"]
    """
    minmax = [[None,None],[None,None],[None,None]]
#    Here we loop through the file, finding the global maximum and minimum of the data set.
#    This is an expensive operation (for HDD time) but optimally it will only have to run once ever
#    so that's how I'm justifying it.
    
    with open(inFileName,'r') as infile:
        first = True
        for line in infile:
            line = line.strip()
            if line[0] != "#":
                row = line.split(',')
                coord = None
                try:
                    coord = (float(row[14]),float(row[15]),float(row[16]))
                except ValueError:
                    pass
                if coord is not None:    
                    if first:
                        minmax[X][MIN] = coord[X]
                        minmax[X][MAX] = coord[X]
                        minmax[Y][MIN] = coord[Y]
                        minmax[Y][MAX] = coord[Y]
                        minmax[Z][MIN] = coord[Z]
                        minmax[Z][MAX] = coord[Z]
                        first = False
                    else:
                        for dimension in range(3):
                            #if the current coordinate is bigger than the current max
                            if coord[dimension]>minmax[dimension][MAX]:
                                #store the coordinate in the max part of the file
                                minmax[dimension][MAX] = coord[dimension]
                            #same ish for the minimums
                            elif coord[dimension]<minmax[dimension][MIN]:
                                minmax[dimension][MIN] = coord[dimension]
   """ """    
#    We now know the size of the box and will be able to use that to decide the chopping points for
#    the galaxies. Who knows if we will ever need any more information out of the box (it's entirely possible)
#    so we are going to copy entire lines.

#    Side note: wouldn't it be cool to store the file as a KD tree in a file?

    Also: be careful of the box's edges because since they are minimums and maximums they refer to an
    ACTUAL COORDINATE and will probably cause inequalities to explode a bit.

    This is actually the hard part that requires a lot of thought. Wish me luck.
    
    macroBoxSize = [dimension[MAX]-dimension[MIN] for dimension in minmax]
    microBoxSize = tuple(macroBoxSize[dimension]/divs[dimension] for dimension in range(3))
    #so first figure out the size in Mpc of the huge box
    #then figure the sizes in Mpc of the small boxes
    #Use that to partition the box into smaller boxes
    boxPartitions = 
    """
    #I would like for the padding that I described in my ipad file to be implemented,

    #note: box 0,0,0 will contain the box with defining corners (0,0,0)Mpc and (xsize,ysize,zsize).
    #all boxes will be inclusive on lower-numbered faces and exclusive on higher-numbered faces

    #We'll build "normal" boxes starting at minimum + expectedRadius
    #We will also build the extended boxes each centered around a normal box
    #These operations will run concurrently so as to minimize unnecessary disk i/o usage

    fileType = inFileName.split('.')[-1].lower()
    fileParms = {'dat':{'x':0,
                        'y':1,
                        'z':2,
                        'split':None
                        },
                 'csv':{'x':14,
                        'y':15,
                        'z':16,
                        'split':','
                        }
                 }
    
    with open(inFileName,'r') as infile:
        for rawline in infile:
            line = rawline.strip()
            if line[0] != "#":
                row = line.split(fileParms[fileType]['split'])
                coord = None
                try:
                    coord = (float(row[fileParms[fileType]['x']]),
                             float(row[fileParms[fileType]['y']]),
                             float(row[fileParms[fileType]['z']]))
                except ValueError:
                    pass
                if coord is not None:
                    boxIndex = (math.floor(coord[0]/sizes[0]),
                                math.floor(coord[1]/sizes[1]),
                                math.floor(coord[2]/sizes[2]))
                    boxfilename = common.getBoxName(outFileName,*boxIndex)
                    #NOTE: The asterisk passes each part of the tuple as one argument.
                    #Which is REALLY HANDY and also REALLY OBSCURE. Be careful!
                    
                    filebounds[boxfilename] = (list(map(lambda x, y: x*y,boxIndex,sizes)),
                                               list(map(lambda x, y: (x+1)*y,boxIndex,sizes)))
                    #calculate the bounding box of this box and add it to a dictionary for later use.
                    with open(boxfilename, 'a') as boxfile:
                        boxfile.write(str(coord[0])+','+str(coord[1])+','+str(coord[2])+'\n')

    genericInfo = {"list_of_files": filebounds,
                   "box_x_size": sizes[0],
                   "box_y_size": sizes[1],
                   "box_z_size": sizes[2]
                  }
    with open(outFileName,'w') as infofile:
        infofile.write(json.dumps(genericInfo,
                                  sort_keys = True,
                                  indent = 4,
                                  separators = (',', ': ')))
    with open(outFileName + '_README','w') as readmefile:
        readmefile.write("""User's guide to the {0} file.
{0} is in JSON format, and contains information about the boxes in this folder.
box_x_size, box_y_size, and box_z_size are all floats that describe the size of each box
in the x, y and z dimensions. 
List_of_files is a dictionary. Its keys are names of boxes. Iterate through all the keys
to make sure you've processed each box. The file paths assume that you are running the
python script from the 'millenium/matplot' folder. 
The values are lists of two lists. The first list tells you the x, y, and z coordinates (in that order)
of the smallest corner of the box. The second list tells you the x, y, and z coordinates in that order
of the largest corner of the box. From there you can figure out everything about the bounding box.""".format(outFileName))
    


if __name__ == "__main__":
    print("This python script does not run standalone. Please use the galaxy.py interface.")
