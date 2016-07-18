import matplotlib
import matplotlib.pyplot as plt
import common
import numpy as np
import scipy.optimize as optimize
import matplotlib.backends.backend_pdf as pdfback

def statsrun(args):

    #When having problems with dividing by zero, we can debug more easily by having execution
    #stop completely when we encounter one, instead of continuing on with only a warning

    np.seterr(all='raise')
    all_settings = common.getdict(args.settings)
    binsize = all_settings["binsize"]
    outputFile = all_settings["output_filename"]
    filename = all_settings["survey_filename"]
    chop = float(all_settings["chop"])
    if all_settings["model_override"] is not None:
        override = common.getdict(all_settings["model_override"])
    else:
        override = None

    if "many" in all_settings and all_settings["many"] == True:
        num_files = all_settings["num_files"]
        for x in range(num_files):
            singlerun(filename.format(x),
                      outputFile.format(x),
                      binsize,
                      chop,
                      override
            )
    else:
        singlerun(filename,outputFile,binsize,chop,override)
        
def genBins(binsize,chop):
    return [x*binsize for x in range(int(chop/binsize)+2)]
    #generates bins for a certain bin size. Stops with the bin that slightly overshoots the chop value
    #Always starts at zero
    
def singlerun(filename,outputFile,binsize,chop,modelOverride=None):
    fig = plt.figure()
    galaxies = common.loadData(filename, dataType = "CF2")
    distances = [galaxy.d for galaxy in galaxies]
    #get a list of all the distances to galaxies. This will let us send it directly to the histogram 

    bins_orig = genBins(binsize,chop)

    #Make a histogram using pylab histogram function.
    n, bins, patches = plt.hist(distances, bins_orig, histtype="stepfilled",label="Galaxy Distribution,\n binsize={:.2f}Mpc".format(binsize))

    #Change visual properties of the histogram
    plt.setp(patches, 'facecolor','g','alpha',0.75)
    robot = chi_sq_solver(bins,n,selection_function)
    if modelOverride is None:
        #If we don't have an existing model to use, we find a best fit and plot it
        #Solve the chi squared optimization for the histogram and selection function
        params = robot.result.x
        #Plot the best fit
        domain = np.arange(0,chop,1)
        model = [selection_function(r,*(robot.result.x)) for r in domain]
        plt.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*(robot.result.x),chisq = robot.result.fun))
        chisq = robot.result.fun
    else:
        #Plot the model given in the settings function instead of calculating a new one
        mo = modelOverride["constants"]
        params = [mo['A'], mo['r_0'], mo['n_1'], mo['n_2']]
        chisq = robot.chi_sq(params)
        domain = np.arange(0,chop,1)
        model = [selection_function(r,*params) for r in domain]
        plt.plot(domain,model, 'k--',linewidth=1.5,label="Model fit: $A = {:.3f}$\n$r_0 = {:.3f}$\n$n_1 = {:.3f}$\n$n_2={:.3f}$\n$\chi^2={chisq:.3f}$".format(*params,chisq = chisq))
        
   
    

    #Add axis labels
    plt.ylabel("Galaxy count")
    plt.xlabel("Distance, Mpc/h")
    plt.title("Distribution of Galaxy Distance")
    plt.legend()
    plt.axis([0,chop,0,500])
    fig2 = plt.figure()
    shellVolume = [common.shellVolCenter(robot.centerbins[i],binsize)  for i in range(len(n))]
    plt.title("Galaxies per Cubic Mpc")
    plt.xlabel("Distance, Mpc/h")
    plt.ylabel("Density, galaxies/(Mpc/h)^3")
    density = [n[i]/shellVolume[i] for i in range(len(n))]
    plt.plot(robot.centerbins,density)
    #Save figure
    with pdfback.PdfPages(outputFile+str(binsize)+'.pdf') as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    if modelOverride is None:
    #Write paramaters to a file for later use.
        common.writedict(outputFile+str(binsize)+'_params.json',{'constants':{'A':params[0],
                                                                              'r_0':params[1],
                                                                              'n_1':params[2],
                                                                              'n_2':params[3]},
                                                                 'info':{'shell_thickness': binsize,
                                                                         'max_radius': chop,
                                                                         'chisq': chisq
                                                                     }
                                                             })
    plt.close('all')

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
                #print(self.centerbins[i],*args)
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
    # if value == 0 and r != 0:
    #     print("Uh-oh, the value was zero!")
    #     print(r,A,r_0,n_1,n_2)
    return value

            
if __name__ == "__main__":
    print("This does not run standalone.")
    
