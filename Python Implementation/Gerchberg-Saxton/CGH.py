from PhaseRetrieval import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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

def main():
    img=np.zeros((1000,1000))
    img=add_point(img,10,480,500)
    img=add_point(img,10,500,480)
    img=add_point(img,10,500,520)
    img=add_point(img,10,520,500)
    
    illum=IlluminPatterns(1000,1000)
    illum.Gaussian(0,1000)
    GS = PhaseRetrieve(img=img,illumination=illum.GetPattern()) #illumination=IlluminPatterns.Gaussian((1000,1000),1,1)

    t=time.time()

    GS.Compute(30)

    print("Ellapsed time: %f seconds.\n",time.time()-t)

    GS.Plot([450,550,450,550])
    #GS.Plot([270,730,270,730])
    plt.show()

if __name__=="__main__":
    main()