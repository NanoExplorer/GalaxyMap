import common
import numpy as np
import scipy.spatial as space
import math
np.seterr(all="raise")
def perturb(args):
    num_acks = 0
    second_order_acks = 0
    num_errs = 0
    hubble_constant = 100
    fractional_error = args.frac_error
    galaxies = common.loadData(args.cf2file,'CF2')
    perturbed_vs = []
    delta_vs = []
    for galaxy in galaxies:
        if abs(galaxy.v) > galaxy.cz/10:
            num_acks += 1
            continue
        if not args.distance:
            distance_modulus = np.log(galaxy.d)
            perturbed_dmod = np.random.normal(distance_modulus,abs(distance_modulus*fractional_error),args.num)
            skewed_distance = np.e ** perturbed_dmod
            dv = hubble_constant * (np.e ** (distance_modulus + distance_modulus*args.frac_error) - galaxy.d)
        else:
            skewed_distance = np.random.normal(galaxy.d,abs(galaxy.d*fractional_error),args.num)
            dv = galaxy.d*fractional_error*hubble_constant
            
        if args.naive or args.distance:
            try:
                velocities = galaxy.cz - hubble_constant * skewed_distance
            except FloatingPointError:
                num_errs += 1
                continue
        else:
            try:
                velocities = galaxy.cz * np.log(galaxy.cz / (hubble_constant * skewed_distance) )
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
    print("Also, {} FloatingPoint errors happened, even after taking out the above galaxies.".format(num_errs))

if __name__ == "__main__":
    arrrghs = common.parseCmdArgs([['cf2file'],['outfile'],['frac_error'],['num'],['-n','--naive'],['-d','--distance']],
                                   ['CF2 survey file to perturb','Output file spec. Must contain exactly one "{}" for use in numbering.', 'Fractional error, determines the standard deviation of the normal distribution used in the distance modulus','Number of perturbed survey files to generate','Use the naive velocity estimator v = cz - H0*d','Don\'t use the distance modulus, just use distance (implies -n)'],
                                  [str,str,float,int,'bool','bool'])
    perturb(arrrghs)

