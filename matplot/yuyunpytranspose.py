import numpy as np

for x in range(100):
    inf="/home/christopher/code/Physics/GalaxyMap/matplot/Yuyu data/CF2-gal-{}.npy".format(x)
    outf=inf+".dat"
    with open(outf,'w') as outfile:
        for line in np.load(inf):
            outfile.write("{} {} {} {} {} {}\n".format(*list(line)))

