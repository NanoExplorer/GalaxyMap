import numpy as np

for x in range(100):
    inf="/home/christopher/code/Physics/millennium/matplot/Yuyu data/CF2-gal-bin5-{}.npy".format(x)
    outf=inf+".dat"
    with open(outf,'w') as outfile:
        for line in np.load(inf):
            outfile.write("{} {} {} {} {} {}\n".format(*list(line)))

