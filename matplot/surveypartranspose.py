import common
import numpy as np
import scipy.spatial as space
import math
from multiprocessing import Pool

def proconesurvey(survey):
    use_dvs = False;
    hubble_constant = 100
    fractional_error = 0.1
    outCF2String = ""
    print("Processing survey {}.".format(survey['name']))
    with open(survey['name'],'r') as csvFile:
        for line in csvFile:
            if line[0] == '#':
                continue
            galaxy=common.MillenniumGalaxy(line)
            center = survey['center']
            rotationMatrix = np.matrix(survey['rot'])
            ontoGalaxy = np.array([galaxy.x-center[0],galaxy.y-center[1],galaxy.z-center[2]])
            #ontoGalaxy is the vector from the survey origin to the galaxy
            rotatedCoord = ontoGalaxy #* rotationMatrix
            x = rotatedCoord.item(0)
            y = rotatedCoord.item(1)
            z = rotatedCoord.item(2)
            rho = space.distance.euclidean(ontoGalaxy,[0,0,0])
            if rho == 0:
                continue
            phi = math.degrees(math.acos(z/rho)) - 90
            theta = math.degrees(math.atan2(y,x))+ 180
            peculiarVel = np.dot(ontoGalaxy,[galaxy.velX,galaxy.velY,galaxy.velZ])/rho
            #posVec = ontoGalaxy/space.distance.euclidean(ontoGalaxy,(0,0,0))
            cf2row = [rho*hubble_constant+peculiarVel,#cz
                      rho,#distance (mpc/h)
                      peculiarVel,#peculiar velocity km/sec
                      rho*hubble_constant*0.2,#dv
                      theta,#longitude degrees - 0 - 360
                      phi]#latitude degrees - -90 - 90
            outCF2String = outCF2String + '{}  {}  {}  {}  {}  {}'.format(*cf2row)
            if use_dvs:
                dvs = np.random.normal(peculiarVel,rho*hubble_constant*0.2,20)
                for x in dvs:
                    outCF2String = outCF2String + '  {}'.format(x)
            outCF2String = outCF2String + '\n'
 
    with open(survey['name'] + '_cf2.txt', 'w') as cf2outfile:
        cf2outfile.write(outCF2String)


def transpose(args):
    print("Loading survey...")
    survey_info = common.getdict(args.survey_file)

    print("Success!")
    with Pool(processes=12) as pool:
        pool.map(proconesurvey,survey_info)
        


if __name__ == "__main__":
    arrrghs = common.parseCmdArgs([['survey_file']],
                                  ['Survey .json file with surveys and their centers'],
                                  [str])

    transpose(arrrghs)
