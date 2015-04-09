import common
import numpy as np
import pylab
import scipy.optimize as optimize
import matplotlib.backends.backend_pdf as pdfback


def mainrun(args):
    fig = pylab.figure()
    all_settings=common.getdict(args.settings)
    filename = all_settings["survey_filename"]
    outputFile = all_settings["output_filename"]
    numBins = all_settings["num_bins"]
    galaxies = common.loadData(filename, dataType = "CF2")
    distances = [galaxy.d for galaxy in galaxies]
    n, bins, patches = pylab.hist(distances, int(numBins), histtype="stepfilled",label="Galaxy Distribution")
    pylab.setp(patches, 'facecolor','g','alpha',0.75)
    robot = chi_sq_solver(bins,n,selection_function)
    print(*(robot.result.x))
    
    model = [selection_function(r,*(robot.result.x)) for r in robot.centerbins]
    pylab.plot(robot.centerbins,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$".format(*(robot.result.x)))
    #pylab.show()
    pylab.ylabel("Galaxy count")
    pylab.xlabel("Distance, Mpc/h")
    pylab.title("Distribution of Galaxy Distance")
    pylab.legend()
    with pdfback.PdfPages(outputFile+str(numBins)) as pdf:
        pdf.savefig(fig)


class chi_sq_solver:
    def __init__(self,bins,ys,function):
        self.bins = bins
        self.ys = ys
        self.centerbins = self.centers(self.bins)
        self.function = function
        self.result = optimize.minimize(self.chi_sq,
                                        np.array([1000,10,0.005,0.005]),
                                        bounds = [(0,None),(0.01,None),(0,None),(0,None)])
        
    def chi_sq(self, args):
        sum = 0
        for i in range(len(self.ys)-1):
            E = self.function(self.centerbins[i], *args)
            sum += (self.ys[i]-E)**2/E
        return sum

    def centers(self,bins):
        centers = []
        for i in range(len(bins)-1):
            centers.append((bins[i+1]-bins[i])/2+bins[i])
        return centers

def selection_function(r, A, r_0, n_1, n_2):
    ratio = r/r_0
    return A*(ratio**n_1)*(1+ratio**(n_1+n_2))**-1
    
if __name__ == "__main__":
    print("This module does not run standalone. Please use galaxy.py interface")
