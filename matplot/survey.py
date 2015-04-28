import common
import numpy as np
import pylab
import scipy.optimize as optimize
import matplotlib.backends.backend_pdf as pdfback

np.seterr(all='raise')
#When having problems with dividing by zero, we can debug more easily by having execution
#stop completely when we encounter one, instead of continuing on with only a warning

def mainrun(args):
    all_settings = common.getdict(args.settings)
    minBins = int(all_settings["min_num_bins"])
    maxBins = int(all_settings["max_num_bins"])
    step = int(all_settings["num_bins_step"])
    outputFile = all_settings["output_filename"]
    filename = all_settings["survey_filename"]
    print("{: >10}{: >10}{: >10}{: >10}{: >10}{: >10}".format("Bins","chi^2","A","r_0","n_1","n_2"))
    for x in range(minBins,maxBins,step):
        #This is how we vary the number of bins
        singlerun(filename,outputFile,x)
        

def singlerun(filename,outputFile,numBins):
    fig = pylab.figure()
    galaxies = common.loadData(filename, dataType = "CF2")
    distances = [galaxy.d for galaxy in galaxies]
    n, bins, patches = pylab.hist(distances, int(numBins), histtype="stepfilled",label="Galaxy Distribution, binsize={:.2f}Mpc".format(max(distances)/int(numBins)))
    pylab.setp(patches, 'facecolor','g','alpha',0.75)
    robot = chi_sq_solver(bins,n,selection_function)
    #print(*(robot.result.x))
    
    model = [selection_function(r,*(robot.result.x)) for r in robot.centerbins]
    pylab.plot(robot.centerbins,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*(robot.result.x),chisq = robot.result.fun))
    print("{: >10}{: >10,.2f}{: >10,.2f}{: >10,.2f}{: >10,.2f}{: >10,.2f}".format(numBins,robot.result.fun,*(robot.result.x)))
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
                                        bounds = [(1,None),(0.01,None),(0,None),(0,None)])
        
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
    value = A*(ratio**n_1)*(1+ratio**(n_1+n_2))**-1
    if value == 0:
        print(r,A,r_0,n_1,n_2)
    return value
    
if __name__ == "__main__":
    print("This does not run standalone.")
