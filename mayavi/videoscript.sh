#!/bin/bash
python mayavi_animated.py
cd /home/christopher/code/Physics/
for i in `seq 0 719` ;
 do montage galaxy$i.png random$i.png -geometry 960x1080+0+0 outpics/out$i.png;
done
cd outpics/
rename 's/out(\d{2})\.png/out0$1\.png/' *.png
rename 's/out(\d{1})\.png/out00$1\.png/' *.png
mencoder mf://*.png -mf fps=25:type=png -ovc x264 -o HDgalaxies.mkv
