import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd


def FourPlot(df, variable, HistogramParams=None,Distribution=None):
    fig, axes = plt.subplots(2, 2)

    fig.suptitle('4-Plot: ' + variable)

    axes[0,0].plot(df[variable].values,'k')
    axes[0,0].set_title('Run Sequence')

    axes[0,1].plot(df[variable].values, df[variable].shift(-1).values,'kx')
    axes[0,1].set_title('Lag Plot')

    if HistogramParams:
        df.hist(bins=np.linspace(HistogramParams[0],HistogramParams[1],HistogramParams[2]),ax=axes[1,0],color='k')
    else:
        df.hist(ax=axes[1,0],color='k')
    axes[1,0].set_title('Histogram')

    if Distribution is None:
        Distribution = 'norm'
    stats.probplot(df[variable].values, plot=axes[1,1],dist = Distribution)
    axes[1,1].set_title('Prob. Plot: '+ Distribution)
    axes[1,1].get_lines()[0].set_marker('x')
    axes[1,1].get_lines()[0].set_markerfacecolor('k')
    axes[1,1].get_lines()[0].set_markeredgecolor('k')

    plt.show()

def AutoCorrelationPlot(df,len):
    N = df.size
    mu = float(df.mean())
    std = float(df.std())
    Conf95 = stats.norm.ppf(1-0.05/2) / np.sqrt(N)
    Conf99 = stats.norm.ppf(1-0.01/2) / np.sqrt(N)

    C0 = np.sum((np.asarray(df.values) - mu)**2) / N
    

    h = np.linspace(0,N-1,N,endpoint = True)
    h = h.astype(int)
    Ch = np.zeros(N)

    for t in range(N):
        Y1 = df.values[t:N-h[t]]
        Y2 = df.values[t+h[t]:N]
        Ch[t] = np.sum(np.multiply(Y1,Y2)) / N

    Rh = Ch / C0
    
    fig,ax = plt.subplots()
    #pd.plotting.autocorrelation_plot(df.values,ax=ax)
    ax.plot(Rh[:len+1],'ko-',label = None)
    ax.plot([0,len+1],[0,0],'k',linewidth = 1,label = None)
    ax.plot([0,len+1],[Conf95,Conf95],'k',linewidth = 1,label = '95% Confidence')
    ax.plot([0,len+1],[-Conf95,-Conf95],'k',linewidth = 1,label = None)
    ax.plot([0,len+1],[Conf99,Conf99],'k--',linewidth = 1,label = '99% Confidence')
    ax.plot([0,len+1],[-Conf99,-Conf99],'k--',linewidth = 1,label = None)
    ax.set_xlabel('lag')
    ax.set_ylabel('Aautocorrelation')

    ax.annotate(
        'Lag 1 = ' + str(round(Rh[1],3)),
        xy = (1,Rh[1]+0.02),
        xytext = (1,Rh[1]+0.3),
        arrowprops=dict(facecolor='black',headwidth=4,width=2,headlength=4),
        horizontalalignment='left',verticalalignment='top'
        
    )

    plt.legend()
    plt.show()