__author__ = 'Elliot I. Simon'
__email__ = 'ellsim@dtu.dk'
__version__ = 'May 16, 2022'

import sys
import os
import glob
from pathlib import Path

lidar = 'Globe3'
inpath = 'R:\\Globe\\Motion_data\\Globe_3'
outpath = 'D:\\ellsim\\Globe\\tiltdata\\Globe3'

yeardirs = glob.glob(inpath + '/*/', recursive = True)
for y in yeardirs:
    monthdirs = glob.glob(y + '/*/', recursive=True)
    for m in monthdirs:
        daydirs = glob.glob(m + '/*/', recursive=True)
        #print(daydirs)
        for d in daydirs:
            print(d)
            os.system("python main.py {0} {1} {2}".format(lidar, d, outpath))