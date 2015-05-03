import scipy.optimize
import matplotlib
matplotlib.use("TkAgg")
import common
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = common.loadData("../tully-fisher.csv", dataType = "miscFloat")
    pairs = [(row[0],np.average(row[1:5])) for row in data]
    xs = [point[1] for point in pairs]
    ys = [point[0] for point in pairs]
    fig = plt.figure(figsize=(8,6),dpi=90)
    ax = fig.add_subplot(111)
    ax.set_title("Tully-Fisher relationship")
    plt.xlabel("Galaxy absolute magnitude")
    plt.ylabel("Galaxy rotational velocity")

    ax.set_yscale("log", nonposx='clip')
    plt.plot(xs, ys, '.')

    
    
    ax.axis([max(xs)+1,min(xs)-1,min(ys)-20,max(ys)+100])
    #print(min(xs),max(xs),min(ys),max(ys))
    plt.show()


if __name__ == "__main__":
    main()


    
