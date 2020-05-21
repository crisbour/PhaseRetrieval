import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv
import numpy as np


def main():
    x=[]
    y=np.loadtxt('Uniformity.txt')
    n=y[0].astype(int)
    y=np.delete(y,0)
    for i in range(n):
        x.append(i);
    plt.plot(x,y,label='Uniformity')
    plt.show()

    

if __name__=="__main__":
    main()
