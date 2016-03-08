import common
import numpy as np
import scipy.spatial as space
import math
np.seterr(all="raise")

def modulusify(dist, args):
    if args.altmodulus:
        return 5*np.log10(dist) + 25
    else:
        return np.log10(dist)
        
def unmodulusify(modulus, args):
    if args.altmodulus:
        return 10**((modulus - 25)/5)
    else:
        return 10 ** modulus


def perturb_distance(distance,error,args):
    if args.distance:
        return np.random.normal(distance,abs(distance*error),args.num)
    else:
        distance_modulus = modulusify(galaxy.d,args)
        if args.unrelative:
            perturbed_dmod = np.random.normal(distance_modulus,fractional_error,args.num)
        else:
            perturbed_dmod = np.random.normal(distance_modulus,abs(distance_modulus*fractional_error),args.num)
            skewed_distance = unmodulusify(perturbed_dmod,args)
        

def perturb(args):
    num_acks = 0
    second_order_acks = 0
    num_errs = 0
    hubble_constant = 100
    fractional_error = args.frac_error
    galaxies = common.loadData(args.cf2file,'CF2')
    perturbed_vs = []
    delta_vs = []
    if args.altmodulus and args.naive:
        print("Altmodulus and Naive are mutually exclusive options.")
        exit()


    for galaxy in galaxies:
        #q_0 = -0.595
        #z = galaxy.cz/(3*10**8)
        #zmod = z*(1 + 0.5*(1-q_0)*z + (1/6)*(2-q_0-3q_0**2)*z**2)
        if abs(galaxy.v) > galaxy.cz/10:
            num_acks += 1
            continue
        if not args.distance:
            
        else:
            skewed_distance = np.random.normal(galaxy.d,abs(galaxy.d*fractional_error),args.num)
            
        if args.naive or args.distance:
            try:
                velocities = galaxy.cz - hubble_constant * skewed_distance
                dv = galaxy.d*fractional_error*hubble_constant
            except FloatingPointError: #I don't think it's possible to have a FP error here... Could be wrong?
                num_errs += 1
                print("I was wrong")
                continue
        else:
            try:
                velocities = galaxy.cz * np.log(galaxy.cz / (hubble_constant * skewed_distance) )
                dv = galaxy.cz*fractional_error#calculate_error(distance_modulus,galaxy.d,frac_error,args)
                for velocity in velocities:
                    if abs(velocity) > galaxy.cz / 10:
                        second_order_acks += 1
            except FloatingPointError:
                num_errs += 1
                continue

        perturbed_vs.append((velocities,galaxy))
        delta_vs.append(dv)

    for n in range(args.num):
        outCF2String = ""
        for i,pv in enumerate(perturbed_vs):
            galaxy = pv[1]
            cf2row = [galaxy.cz,
                      (galaxy.cz - pv[0][n])/100,
                      pv[0][n],
                      delta_vs[i],
                      galaxy.lon,
                      galaxy.lat]
            outCF2String = outCF2String + '{}  {}  {}  {}  {}  {}\n'.format(*cf2row)

        with open(args.outfile.format(n), 'w') as cf2outfile:
            cf2outfile.write(outCF2String)

    print("{} out of {} galaxies ({:.2f}) had true velocity NOT much less than redshift,".format(num_acks,len(galaxies),num_acks/len(galaxies)))
    print("i.e. the condition on our estimator that v << cz was not satisfied.")
    print("This happened to the random data {} times out of {}.".format(second_order_acks,args.num*len(galaxies)))
    print()
    print("Also, {} FloatingPoint errors happened, even after taking out the close-by galaxies.".format(num_errs))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('cf2file',help='CF2 survey file to perturb',type=str)
    parser.add_argument( 'outfile' ,help= 'Output file spec. Must contain exactly one "{}" for use in numbering.' ,type=str)
    parser.add_argument('frac_error',help='Fractional error, determines the standard deviation of the normal distribution used in the distance modulus',type=float)
    parser.add_argument('num',help="Numper of perturbed survey files to generate",type=int)
    parser.add_argument('type',choices=['distance','modulus','relative'],help="The type of perturbations, consistent naming with my paper")
    parser.add_argument('estimator',choices=['cz','feldman'])
    parser.add_argument('modulus',choices=['ln','log','textbook'])

    arrrghs = parser.parse_args()
    perturb(arrrghs)


#You need to specify a place (perturbation type), a modulus, and an estimator.
