import common
from multiprocessing import Pool
from scipy.spatial import cKDTree
import numpy as np


def prune(filename):
    with open(filename,'r') as boxfile:
        data = boxfile.readlines()
    dataCopy = [line for line in data]
    #DON'T MODIFY THE DATACOPY!
    
    isComment = True
    offset = -1
    while isComment:
        offset += 1
        #print(offset,len(data))
        isComment = (data[offset][0] == '#') or (data[offset][0] == 'x')
        #Note, after this script is done processing a millennium file, the offset method will no longer work
        #because there will be comments everywhere in the body of the csv file.
    positions = common.loadData(filename,'millPos')
    kd = cKDTree(positions)
    pairs = kd.query_pairs(0.0001)
    for g1,g2 in pairs:
        #build a new galaxy, with averaged things, but summed mass
        #print(data[g1+offset])
        #print(data[g2+offset])
        gal1 = common.MillenniumGalaxy(dataCopy[g1+offset])
        gal2 = common.MillenniumGalaxy(dataCopy[g2+offset])
        totalMass = gal1.mvir + gal2.mvir
        weightedVelocity1 = np.array([gal1.velX,gal1.velY,gal1.velZ])*gal1.mvir
        weightedVelocity2 = np.array([gal2.velX,gal2.velY,gal2.velZ])*gal2.mvir
        averageGalaxy = common.MillenniumGalaxy([gal1.x,
                                                 gal1.y,
                                                 gal1.z,
                                                 (gal1.velX + gal2.velX) / 2,
                                                 (gal1.velY + gal2.velY) / 2,
                                                 (gal1.velZ + gal2.velZ) / 2,
                                                 totalMass,
                                                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        data[g1+offset] = '#REMOVED BECAUSE IT IS A DUPLICATE#'+dataCopy[g1+offset]
        data[g2+offset] = '#REMOVED BECAUSE IT IS A DUPLICATE#'+dataCopy[g2+offset]
        data.append('#AVERAGE GALAXY FOLLOWS#\n')
        data.append(averageGalaxy.toString()+'\n')
        data.append('#The above galaxy was added as an average of two galaxies that were in the same location\n')
        data.append('#The only information it has associated with it is POSITION, VELOCITY, and VIRIAL MASS, because none of the other attributes were currently in use (and I didn\'t know how to use them) when this file was created.\n')
    with open(filename,'w') as newFile:
        data = newFile.writelines(data)
    return len(pairs)        


filesNames = '/home/christopher/code/Physics/GalaxyMap/matplot/surveys/100s/CF2_gal{}.mil'
numFiles = 100
offset = 0

listOfFiles = [filesNames.format(n+offset) for n in range(numFiles)]

pool = Pool()

count = pool.map(prune, listOfFiles)

print('There were a total of {} close binaries'.format(sum(count)))
