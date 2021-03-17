import csv
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import stats
import numpy as np
import statistics as st
import math
from mpl_toolkits import mplot3d
from matplotlib import rc
from pylab import MaxNLocator
import pandas as pd
import datetime as dt
import NISTplots as nist

#open the data file
df = pd.read_csv('RANDN.DAT', skiprows=range(0,25),header=None,delim_whitespace=True)
#convert the matrix of random numbers into 1 long list
ds=pd.Series(df.values.ravel('F'))
df = pd.DataFrame(ds,columns = ['Random Numbers'])

#create the four plot for the data
nist.FourPlot(df,'Random Numbers')

