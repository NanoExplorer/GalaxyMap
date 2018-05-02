import matplotlib
matplotlib.use("TkAgg")
import common
import numpy as np
import matplotlib.backends.backend_pdf as pdfback
import math
import matplotlib.pyplot as plt
import scipy.optimize

def updateMinMax(curmin, curmax, values):
    for x in values:
        if x > curmax:
            curmax = x
        if x < curmin:
            curmin = x
    return (curmin, curmax)
#NYI
# def jackknife(allys):
#     """ This method calculates the error using the jackknife method, an explanation of which
#     can be found at:
#     http://www.physics.utah.edu/~detar/phycs6730/handouts/jackknife/jackknife/
#     """
#     jk = []
#     for ys in allys:        
#         jk.append((len(ys)-1)*sum([(2*np.std("""All but one""")-]))
    
def statistics(args):
    data = common.getdict(args.datafile)
    #Create a new figure. Each page in the pdf file is a figure
    fig = plt.figure(1,figsize=(8,6),dpi=400)
    #Create a subplot axis
    ax = fig.add_subplot(111)
    #Set the title, x and y labels of this page's plot.
    ax.set_title("Correlation Function, Davis and Peebles Estimator")
    plt.xlabel("Correlation distance, Mpc/h")
    plt.ylabel("Correlation")

    fig2 = plt.figure(2,figsize=(8,6),dpi=400)
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Correlation Function, Hamilton Estimator")
    plt.xlabel("Correlation distance, Mpc/h")
    plt.ylabel("Correlation")

    fig3 = plt.figure(3,figsize=(8,6),dpi=400)
    ax3 = fig3.add_subplot(111)
    plt.ylabel("Correlation")
    plt.xlabel("Correlation distance, Mpc/h")
    ax3.set_title("Correlation Function, Landy and Szalay Estimator")

    fig4 = plt.figure(4,figsize=(8,6),dpi=400)
    ax4 = fig4.add_subplot(111)
    plt.ylabel("Correlation+1")
    plt.xlabel("Correlation distance, Mpc/h")
    ax4.set_title("Correlation Function from random points, modified Davis and Peebles Estimator")

    fig5 = plt.figure(5,figsize=(8,6),dpi=400)
    ax5 = fig5.add_subplot(111)
    plt.ylabel("Correlation")
    plt.xlabel("Correlation distance, Mpc/h")
    ax5.set_title("Correlation Function, average. Landy and Szalay estimator.")
    ax5.set_xscale('log', nonposx = 'clip')
    ax5.set_yscale('log', nonposy = 'clip')

    fig6 = plt.figure(6,figsize=(8,6),dpi=400)
    ax6 = fig6.add_subplot(111)
    plt.ylabel("Correlation Residuals")
    plt.xlabel("Correlation distance, Mpc/h")
    ax6.set_title("Correlation Function residuals")
    ax6.set_xscale('log', nonposx = 'clip')
    #ax6.set_yscale('log', nonposx = 'clip')

    #Set all the axes to log scale
    
    
    ax.set_xscale("log", nonposx='clip')
    ax2.set_xscale("log", nonposx='clip')
    ax3.set_xscale("log", nonposx='clip')
    ax4.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax2.set_yscale("log", nonposy='clip')
    ax3.set_yscale("log", nonposy='clip')
    ax4.set_yscale("log", nonposy='clip')
    ys = []
    maxY =10**1
    minY = 10**-2
    maxRandom = 5*10**1
    minRandom = 10**-1
    numboxes = 0
    for box in data['raw_runs'][0].items():
        #The .items() function returns a tuple (Key, value)
        #That's why there are all the box[1]'s running around.
        if box[0] != "ALL_BOXES":
            #Each box has its own data associated with it, so first we plot ALL the data
            plt.figure(1)
            plt.plot(box[1]["rs"],box[1]["Davis_Peebles"],'.')
            plt.figure(2)
            plt.plot(box[1]["rs"],box[1]["Hamilton"],'.')
            plt.figure(3)
            plt.plot(box[1]["rs"],box[1]["Landy_Szalay"],'.')
            plt.figure(4)
            plt.plot(box[1]["rs"],[x+1 for x in box[1]["Random_Correlation"]],'.')
            ys.append(box[1]["Landy_Szalay"])
            #minY, maxY = updateMinMax(minY, maxY, box[1]["Davis_Peebles"])
            #minY, maxY = updateMinMax(minY, maxY, box[1]["Hamilton"])
            #minY, maxY = updateMinMax(minY, maxY, box[1]["Landy_Szalay"])
            #minRandom, maxRandom = updateMinMax(minRandom, maxRandom, box[1]["Random_Correlation"])
            #This was an attempt to give all of the graphs the same scales. I don't know why it didn't work...
            numboxes += 1
            #Here we count the number of boxes so that we know whether we can use the standard deviation
            #for error bars
    power = lambda r, r0, gamma: (r/r0)**(-gamma) 
    #power law for estimating correlation and its relation to distance.
    #Used in the curvefit scipy function

    allys = list(zip(*ys))
    #This list contains tuples of y-values for a certain x value for use in calculating the standard
    #deviation easily. Format [(10,9.8,10.25),(7.776,7.90,7.745) etc] except with possibly more values per tuple
    #and definitely way more tuples.
    
    
    #Calculate the 95% confidence interval, two times the standard deviation of all the ys for a certain x.

    #yerrs = jackknife(allys)
    
    ys = data['raw_runs'][0]["ALL_BOXES"]["Landy_Szalay"]
    xs = data['raw_runs'][0]["ALL_BOXES"]["rs"]
    xerrs = [data['raw_runs'][0]["ALL_BOXES"]["dr_left"],data['raw_runs'][0]["ALL_BOXES"]["dr_right"]]
    #Take the raw xs and ys from the dataset that was averaged over all of the boxes.
    if numboxes == 1:
        popt, pcov = scipy.optimize.curve_fit(power,xs,ys,p0=(10,1.5))#,sigma=yerrs,absolute_sigma=True)
        #When we only have one box, we need to tell the curve fit that all of the errors are "The Same"
        yerrs = [300*ys[i]*math.sqrt(data['raw_runs'][0]["ALL_BOXES"]["DDs"][i])/data['raw_runs'][0]["ALL_BOXES"]["DDs"][i]for i in range(len(ys))]
    else:
        yerrs = [np.std(y) for y in allys]
        popt, pcov = scipy.optimize.curve_fit(power,xs,ys,p0=(10,1.5),sigma=yerrs,absolute_sigma=True)
        #More than one box means that the standard deviation errors are correct.
        
    print(yerrs)
    # print(pcov)
    # print(popt)
    plt.figure(5)
    dot = plt.errorbar(xs,ys,yerr=yerrs,xerr=xerrs,fmt='.',label="Averaged Correlation Data")
    model = [power(x,*popt) for x in xs]
    line = plt.plot(xs,
                    model,
                    label="Model fit: $(r/r_0)^{{-\gamma}}$\n$r_0 = {:.3f}$\n$\gamma = {:.3f}$".format(popt[0],
                                                                                                       popt[1]))
    #We need {{ and }} to escape the .format thingy and pass { and } to LaTeX

    plt.legend()
    plt.figure(6)
    residuals = [y/mod for y,mod in zip(ys,model)]

    #Since a residual data point is y / model, the relative error in residual will be equal to
    #sqrt( relative error in model ^2 + relative error in point ^2)
    residuals_errors = [res*(dy / y) for y,dy,res in zip(ys,yerrs,residuals)]
    plt.errorbar(xs,residuals,yerr=residuals_errors,fmt='.',label="Residuals")
    plt.plot(xs,[1 for x in xs],label="Model")
    plt.legend()
    ax6.axis([0,max(xs)+1,0,2])#min(residuals)*.9,max(residuals)*1.1])
    plt.figure(5)
    #Here, we set the scale of each axis. We need a better method of dynamically deciding what the bounds should be
    ax5.axis([min(xs)-0.15,max(xs)+3,minY,maxY])
    plt.figure(4)
    ax4.axis([min(xs)-0.15,max(xs)+3,minRandom,maxRandom])
    plt.figure(3)
    ax3.axis([min(xs)-0.15,max(xs)+3,minY,maxY])
    plt.figure(2)
    ax2.axis([min(xs)-0.15,max(xs)+3,minY,maxY])
    plt.figure(1)
    ax.axis([min(xs)-0.15,max(xs)+3,minY,maxY])
    #plt.legend([dot,line],["Data","Best fit"])
    basefilename = args.datafile.replace("rawdata.json","")
    with open("output/statistics.csv", 'a') as outfile:#WARNING:: I usually don't like using static file paths!!!!!!!!
        #ANOTHER WARNING: every time you run stats it appends a line to this file. So, be careful and only use the
        #statistics file after cleaning it and doing a very controlled run.
        line = ""
        for datapt in [float(x) for x in popt]:
            line = line + str(datapt)+","
        for error in np.sqrt(np.diag(pcov)):
            line = line + str(error)+','
        line = line + str(data["settings"]["Divide"]["x_box_size"]*
                          data["settings"]["Divide"]["y_box_size"]*
                          data["settings"]["Divide"]["z_box_size"])
        outfile.write(line+'\n')              
    with pdfback.PdfPages(basefilename+'graphs.pdf') as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        pdf.savefig(fig4)
        pdf.savefig(fig5)
        pdf.savefig(fig6)


if __name__ == "__main__":
    print("This script should be activated using 'galaxy.py stats inputfile'. It does not run standalone.")
