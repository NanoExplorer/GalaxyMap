import common
import numpy as np
import pylab
import scipy.optimize as optimize
import matplotlib.backends.backend_pdf as pdfback

np.seterr(all='raise')
#When having problems with dividing by zero, we can debug more easily by having execution
#stop completely when we encounter one, instead of continuing on with only a warning

def statsrun(args):
    all_settings = common.getdict(args.settings)
    binsize = int(all_settings["binsize"])
    outputFile = all_settings["output_filename"]
    filename = all_settings["survey_filename"]
    chop = float(all_settings["chop"])

    singlerun(filename,outputFile,binsize,chop)
        
def genBins(binsize,chop):
    return [x*binsize for x in range(int(chop/binsize)+2)]
    #generates bins for a certain bin size. Stops with the bin that slightly overshoots the chop value
    #Always starts at zero
    
def singlerun(filename,outputFile,binsize,chop):
    fig = pylab.figure()
    galaxies = common.loadData(filename, dataType = "CF2")
    distances = [galaxy.d for galaxy in galaxies]
    #get a list of all the distances to galaxies. This will let us send it directly to the histogram function

    # for i in range(len(distances)-1,-1,-1):
    #     #loop BACKWARDS through the array and get rid of entries bigger than the chop value.
    #     #Why are we doing this? Does it actually affect the results positively?
    #     #Answer: it appears this was an artifact of a previous chop implementation.
    #     if distances[i] > chop:
    #         del distances[i]
    
    bins_orig = genBins(binsize,chop)

    #Make a histogram using pylab histogram function.
    n, bins, patches = pylab.hist(distances, bins_orig, histtype="stepfilled",label="Galaxy Distribution,\n binsize={:.2f}Mpc".format(binsize))

    #Change visual properties of the histogram
    pylab.setp(patches, 'facecolor','g','alpha',0.75)

    #Solve the chi squared optimization for the histogram and selection function
    robot = chi_sq_solver(bins,n,selection_function)

    #Plot the best fit
    domain = np.arange(0,chop,1)
    model = [selection_function(r,*(robot.result.x)) for r in domain]
    pylab.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*(robot.result.x),chisq = robot.result.fun))

    #Add axis labels
    pylab.ylabel("Galaxy count")
    pylab.xlabel("Distance, Mpc/h")
    pylab.title("Distribution of Galaxy Distance")
    pylab.legend()

    #Save figure
    with pdfback.PdfPages(outputFile+str(binsize)) as pdf:
        pdf.savefig(fig)
    params = robot.result.x
    #Write paramaters to a file for later use.
    common.writedict(outputFile+str(binsize)+'_params.json',{'A':params[0],
                                                             'r_0':params[1],
                                                             'n_1':params[2],
                                                             'n_2':params[3]
                                                            })

class chi_sq_solver:
    def __init__(self,bins,ys,function):
        #Initialize object members
        self.bins = bins
        self.ys = ys
        self.centerbins = self.centers(self.bins)
        self.function = function
        #Calculate minimum chi squared fit
        self.result = optimize.minimize(self.chi_sq,
                                        np.array([1000,10,0.005,0.005]),
                                        bounds = [(1,None),(0.01,None),(0,20),(0,20)])
        
    def chi_sq(self, args):
        #Return the chi_squared statistic for the binned data and the arguments for the function
        sum = 0
        for i in range(len(self.ys)-1):
            try:
                E = self.function(self.centerbins[i], *args)
            except FloatingPointError:
                print("Oh no! There was a floating point error.")
                print(self.centerbins[i],*args)
                exit()
            sum += (self.ys[i]-E)**2/E
        return sum

    def centers(self,bins):
        #Calculate the centers of the bins, or the average x value of each bin.
        #Would be mathematically cool if we could average the x values of all the data points,
        #but that probably shouldn't affect anything substantially.
        centers = []
        for i in range(len(bins)-1):
            centers.append((bins[i+1]-bins[i])/2+bins[i])
            #Take the average value of two bins and add it to the lower bin value
        return centers

def selection_function(r, A, r_0, n_1, n_2):
    ratio = r/r_0
    #Selection function as defined by what I got in my email from Professor Feldman
    value = A*(ratio**n_1)*(1+ratio**(n_1+n_2))**-1
    if value == 0 and r != 0:
        print("Uh-oh, the value was zero!")
        print(r,A,r_0,n_1,n_2)
    return value
    
if __name__ == "__main__":
    print("This does not run standalone.")
