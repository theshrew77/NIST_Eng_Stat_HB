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
import sys


sys.path.insert(1, '/NIST_Eng_Stat_HB//Modules')
import NISTplots as nistplt
import NISTtables as nisttbl


#from /../../Modules import NISTplots as nist
Variable = 'Random Numbers'
#open the data file
df = pd.read_csv('RANDN.DAT', skiprows=range(0,25),header=None,delim_whitespace=True)

#convert the matrix of random numbers into 1 long list
temp = np.asarray(df.values)
temp = np.reshape(temp,(1,500))
#ds=pd.Series(df.values.ravel('F'))
ds = pd.Series(temp[0])
df = pd.DataFrame(ds,columns = [Variable])
#print(df.head())



#print common summary statistics
nisttbl.SummaryStatsTable(df,Variable)

#print data linear fit parameters
nisttbl.LinearParametersTable(df,Variable)

#create the four plot for the data
#nistplt.FourPlot(df,'Random Numbers',(-10,10,50))

#run a Bartlett's test for equal variance
nisttbl.BartlettsTestTable(df)

#create an autocorrelation plot
#nistplt.AutoCorrelationPlot(df,21)

#run a runs tests for randomness
nisttbl.RunsTestTable(df,Variable)



