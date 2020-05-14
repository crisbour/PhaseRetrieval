from PhaseRetrieve import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator

def add_point(img,diam,i_pos,j_pos):
    '''
    assert(len(diam)==len(i_pos))
    assert(len(i_pos)==len(j_pos))
    '''
    mid=(int)(diam/2)
    i_int=i_pos-mid+np.arange(diam)
    j_int=j_pos-mid+np.arange(diam)

    img[np.ix_(i_int, j_int)]=255*np.ones((diam,diam))
    
    return img

if __name__=="__main__":
    img=np.zeros((1000,1000))
    img=add_point(img,10,480,500)
    img=add_point(img,10,500,480)
    img=add_point(img,10,500,520)
    img=add_point(img,10,520,500)

    t=time.time()

    PR=SLM_Phase(img,AlgorithmType.GS)
    # PR.Compute()



    # print("Ellapsed time: %f seconds.\n",time.time()-t)

    # PR.Plot([470,530,470,530])

    #PR.SetAlgorithm(AlgorithmType.MRAF)
    PR.Compute()

    PR.Plot([470,530,470,530])

    fig2=plt.figure(2)
    u_plt=plt.plot(PR.uniformity)
    fig2.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Itterations')
    plt.ylabel('u')
    plt.title('Uniformity')
    plt.show()
