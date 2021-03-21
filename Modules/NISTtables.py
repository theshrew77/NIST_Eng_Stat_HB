import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from tabulate import tabulate
import pandas as pd



def SummaryStatsTable(df,Variable):

    data = [

        ['Sample Size',df[Variable].size],
        ['Mean',df[Variable].mean()],
        ['Median',df[Variable].median()],
        ['Minimum',df[Variable].min()],
        ['Maximum',df[Variable].max()],
        ['Range',df[Variable].max() - df[Variable].min()],
        ['Stan. Dev.',df[Variable].std()]
    ]

    print('\nSummary Stats Table')
    print(tabulate(data))

def LinearParametersTable(df,Variable):
    P = np.polyfit(df.index+1,df[Variable],1)
    p = np.polyval(P,df.index+1)

    syx =  np.sqrt( sum((df[Variable].values - p)**2) / df.size )
    xbar = np.mean(np.asarray(df.index+1))
    sx = np.sqrt( sum( (np.asarray(df.index+1) - xbar)**2) / df.size )
    t = (P[0] - 0)*sx * np.sqrt(df.size - 2) / syx

    N = df.size
    alpha = 0.05
    
    CriticalValue = max(stats.t.ppf([alpha/2,1-alpha/2],N-1))
    

    data = [
        ['B0',P[1],'tbd','tbd',CriticalValue],
        ['B1',P[0],'tbd',round(t,4),CriticalValue]

    ]


    print('\nLinear Parameters Table')
    print(tabulate(data,headers=['Coefficient', 'Estimate', 'Stan. Error', 't-Value','Critical Value']))



def BartlettsTestTable(df):
    dfs = np.array_split(df,4)
    alpha = 0.05
    k = 4

    #for i in range(len(dfs)):
    #    print(dfs[i].head())
    
    T,P = stats.bartlett(
        dfs[0].values,
        dfs[1].values,
        dfs[2].values,
        dfs[3].values
    )


    P = stats.chi2.ppf(1-alpha,k-1)

    data = [
            ['Test Statistic: T=',T],
            ['Degrees of Freedom: k - 1 = ',k-1],
            ['Significance Level: ', alpha],
            ['Critical value: ',P],
            ['Critical Region: Reject H0 if T > ',P]

        ]
    print('\nBartlett\'s Test Table')
    print(tabulate(data))
 