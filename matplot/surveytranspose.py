import common
import numpy as np
import scipy.spatial as space
import math

def transpose(args):
    hubble_constant = 100
    survey_info = common.getdict(args.survey_file)
    for survey in survey_info:
        outCF2String = "" 
        with open(survey['name'],'r') as csvFile:
            for line in csvFile:
                if line[0] == '#':
                    continue
                galaxy=common.MillenniumGalaxy(line)
                center = survey['center']
                rotationMatrix = np.matrix(survey['rot'])
                ontoGalaxy = np.array([galaxy.x-center[0],galaxy.y-center[1],galaxy.z-center[2]])
                #ontoGalaxy is the vector from the survey origin to the galaxy
                rotatedCoord = ontoGalaxy * rotationMatrix
                x = rotatedCoord.item(0)
                y = rotatedCoord.item(1)
                z = rotatedCoord.item(2)
                rho = space.distance.euclidean(ontoGalaxy,[0,0,0])
                phi = math.degrees(math.acos(z/rho)) - 90
                theta = math.degrees(math.atan2(y,x))+ 180
                peculiarVel = np.dot(ontoGalaxy,[galaxy.velX,galaxy.velY,galaxy.velZ])/rho
                #posVec = ontoGalaxy/space.distance.euclidean(ontoGalaxy,(0,0,0))
                cf2row = [rho*hubble_constant+peculiarVel,#cz
                          rho,#distance (mpc/h)
                          peculiarVel,#peculiar velocity km/sec
                          peculiarVel*0.1,#dv - currently 10%
                          theta,#longitude degrees - 0 - 360
                          phi]#latitude degrees - -90 - 90
                outCF2String = outCF2String + '{}  {}  {}  {}  {}  {}\n'.format(*cf2row)
        with open(survey['name'] + '_cf2.txt', 'w') as cf2outfile:
            cf2outfile.write(outCF2String)
