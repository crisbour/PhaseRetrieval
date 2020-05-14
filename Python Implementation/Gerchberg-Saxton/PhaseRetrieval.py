import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class LoopBlocks:                                   #Functionalitis of blocks found in the generic GS
    def SLMToObj(self,U_SLM):                        #Fourier Transform
        U_f= np.fft.fftshift(np.fft.fft2(U_SLM))
        return U_f
    def ObjToSLM(self,g_f):                        #Inverse Fourier Transform
        g_SLM=np.fft.ifft2(np.fft.ifftshift(g_f))
        return g_SLM
    def ComposeSignal(self,amplitude,phase):
        signal = amplitude*np.exp(1j*phase)
        return signal
    def DecomposeSignal(self,signal):
        amplitude=np.abs(signal)
        phase=np.angle(signal)
        return (amplitude,phase)
    def Unity(self,signal):
        phase=np.angle(signal)
        return signal

class IlluminPatterns:
    def __init__(self,height,width):
        self.height=height
        self.width=width
    def Gaussian(self,miu,sigma):
        x,y = np.meshgrid(range(self.height),range(self.width))
        d=np.sqrt((x-self.height/2)**2+(y-self.width/2)**2)
        self.illum=255*np.exp(-d**2/(2.0*sigma**2))
    def GetPattern(self):
        return self.illum
    

class PhaseRetrieve(LoopBlocks):
    def __init__(self,img=None,illumination=1):
        if img.any()!= None:
            self.SetPattern(img)
        self.SetIllimunation(illumination) 

    def SetIllimunation(self,illum):
        self.illum=illum

    #Set the desired pattern to be achieved at the image plane
    def SetPattern(self,img):
        self.target_amp=np.sqrt(img)
        self.height,self.width=img.shape
        self.phase_slm=2*np.pi*np.random.rand(self.height,self.width)
        self.phase_obj=2*np.pi*np.random.rand(self.height,self.width)
        self.output=np.empty_like(img)

    def Compute(self,niter=10):
        g_f=np.empty((self.height,self.width))
        g_SLM=np.empty((self.height,self.width))
        U_SLM=np.empty((self.height,self.width))
        U_f=np.empty((self.height,self.width))
        _=None
        
        for num_iter in range(niter):
            g_f=self.ComposeSignal(self.target_amp,self.phase_obj)
            g_SLM=self.ObjToSLM(g_f)
            _,self.phase_slm=self.DecomposeSignal(g_SLM)
            U_SLM=self.ComposeSignal(self.illum,self.phase_slm)
            U_f=self.SLMToObj(U_SLM)
            self.output,self.phase_obj=self.DecomposeSignal(U_f)
            
        self.output=255*self.output/self.output.max()

    def ConstructedPattern(self):
        return self.output

    def Plot(self,frame=None):

        focus_region=np.full(self.target_amp.shape,True)
        if frame!=None:
            focus_region=np.ix_(range(frame[0],frame[1]),range(frame[2],frame[3]))

        plt.figure(1)
        plt.subplot(221)
        try:
            plt.imshow(self.illum)
        except:
            illum=np.ones((self.height,self.width))
            plt.imshow(illum)
        plt.title('Illumination')

        plt.subplot(222)
        plt.imshow(self.phase_slm)
        plt.title('Phase mask')

        plt.subplot(223)
        plt.imshow(self.target_amp[focus_region])
        plt.title('Zoomed desired')
        plt.subplot(224)
        
        recovered   =   self.ConstructedPattern()

        plt.imshow(recovered[focus_region])
        plt.title('Zoomed recovered')