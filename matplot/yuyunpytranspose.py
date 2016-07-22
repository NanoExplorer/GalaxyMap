import numpy as np

for x in range(100):
    inf="/data/yuyuw/Millennium/SimulBox/CF2-box/CF2-gal-{}.npy".format(x)
    outf="/data/c156r133/surveys/yuyu/CF2-gal-{}.dat".format(x)
    with open(outf,'w') as outfile:
        for line in np.load(inf):
            outfile.write("{} {} {} {} {} {}\n".format(*list(line)))

