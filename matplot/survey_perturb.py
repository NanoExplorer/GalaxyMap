import common
import numpy as np
import argparse
#import scipy.spatial as space
#import math
np.seterr(all="raise")

def modulusify(dist, args):
    if args == "textbook":
        return 5*np.log10(dist) + 25
    elif args == "log":
        return np.log10(dist)
    elif args == "ln":
        return np.log(dist)
        
def unmodulusify(modulus, args):
    if args == "textbook":
        return 10**((modulus - 25)/5)
    elif args == "log":
        return 10 ** modulus
    elif args == "ln":
        return np.exp(modulus)


def perturb(infile,outfile,err,num,ptype,est,mod,lots):
    """Infile and outfile are filenames. Outfile always has a {} in it, and infile should have one too if you're
    using the 'lots' option.

    err is the error, usually like 0.2 for distance perturbations or 0.02 for modulus perturbations

    ptype, est, and mod are strings that describe the combination of perturbation type, modulus and estimator to use.

    ptype shouls be 'distance', 'modulus', or 'relative'
    est should be 'cz' or 'feldman' where cz=v+h*d and feldman is the unbiased log estimator.
    modulus only applies when using modulus and relative ptypes, and has choices 'ln','log', and 'textbook'
    
    if lots is not 0, False, or None then it should be an integer specifying the number of infiles. infile should
    then have a {} in it for the index. Outfiles will be numbered as follows: infile_index*num + outfile_index
    where infile_index goes from zero to lots and outfile_index goes from 0 to num
    """
    if lots:
        infiles = [infile.format(x) for x in range(lots)]   
    else:
        infiles = [infile]
    for in_i,infile in enumerate(infiles):
        num_acks = 0
        second_order_acks = 0
        num_errs = 0
        hubble_constant = 100
        galaxies = common.loadData(infile,'CF2')
        perturbed_vs = []
        delta_vs = []

        for galaxy in galaxies:
            #q_0 = -0.595
            #z = galaxy.cz/(3*10**8)
            #zmod = z*(1 + 0.5*(1-q_0)*z + (1/6)*(2-q_0-3q_0**2)*z**2)
            if abs(galaxy.v) > galaxy.cz/10:
                num_acks += 1

            if ptype == "distance":
                skewed_distance = np.random.normal(galaxy.d,abs(galaxy.d*err),num)
            elif ptype == "modulus":
                inmod = modulusify(galaxy.d, mod)
                pmod = np.random.normal(inmod,err,num)
                skewed_distance = unmodulusify(pmod, mod)
            elif ptype == "relative":
                inmod = modulusify(galaxy.d,mod)
                pmod = np.random.normal(inmod,np.abs(err*inmod),num)
                skewed_distance = unmodulusify(pmod,mod)
                
            if est == "cz":
                try:
                    velocities = galaxy.cz - hubble_constant * skewed_distance
                    dv = galaxy.d*err*hubble_constant
                except FloatingPointError: #I don't think it's possible to have a FP error here... Could be wrong?
                    num_errs += 1
                    print("I was wrong")
                    continue
            elif est == "feldman":
                try:
                    velocities = galaxy.cz * np.log(galaxy.cz / (hubble_constant * skewed_distance) )
                    dv = galaxy.cz*err#calculate_error(distance_modulus,galaxy.d,frac_error,args)
                    for velocity in velocities:
                        if abs(velocity) > galaxy.cz / 10:
                            second_order_acks += 1
                except FloatingPointError:
                    num_errs += 1
                    continue
            perturbed_vs.append((velocities,dv,skewed_distance,galaxy))

        print("{} out of {} galaxies ({:.2f}) had true velocity NOT much less than redshift,".format(num_acks,len(galaxies),num_acks/len(galaxies)))
        print("i.e. the condition on our estimator that v << cz was not satisfied.")
        print("This happened to the random data {} times out of {}.".format(second_order_acks,num*len(galaxies)))
        print("Also, {} FloatingPoint errors happened, even after taking out the close-by galaxies.".format(num_errs))
        print()
        galaxies = []
        for v,dv,d,galaxy in perturbed_vs:
            galaxies.append(np.array((galaxy.x /galaxy.d,
                             galaxy.y /galaxy.d,
                             galaxy.z /galaxy.d) + galaxy.getRedshiftXYZ()))
        galaxiesnp = np.array(galaxies)
        surveys = []
        for n in range(num):
            survey = [[],[],[]]
            for pv in perturbed_vs:
                survey[0].append(pv[2][n])
                survey[1].append(pv[0][n])
                survey[2].append(pv[1])
        
            surveys.append(survey)

        surveysnp = np.array(surveys)
        print(galaxiesnp.shape)
        print(surveysnp.shape)
        np.save(outfile+"{}_1.npy".format(in_i),galaxiesnp)
        np.save(outfile+"{}_2.npy".format(in_i),surveysnp)
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('cf2file',help='CF2 survey file to perturb',type=str)
    parser.add_argument( 'outfile' ,help= 'Output file spec.' ,type=str)
    parser.add_argument('frac_error',help='Fractional error, determines the standard deviation of the normal distribution used in the distance modulus',type=float)
    parser.add_argument('num',help="Numper of perturbed survey files to generate",type=int)
    parser.add_argument('-n','--numin',type=int,help='Specify the number of input files. If using this option, include one "{}" in the cf2file for the index')
    parser.add_argument('perturbtype',choices=['distance','modulus','relative'],help="The type of perturbations, consistent naming with my paper")
    parser.add_argument('estimator',choices=['cz','feldman'])
    parser.add_argument('modulus',choices=['ln','log','textbook'])

    arrrghs = parser.parse_args()
    perturb(arrrghs.cf2file,
            arrrghs.outfile,
            arrrghs.frac_error,
            arrrghs.num,
            arrrghs.perturbtype,
            arrrghs.estimator,
            arrrghs.modulus,
            arrrghs.numin
    )


#You need to specify a place (perturbation type), a modulus, and an estimator.
