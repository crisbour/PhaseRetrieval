import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# Choose from one of the following Phase Retrieval Alogrithm and pass it to FactoryMethod
class AlgorithmType:
    GS      =   'GerchbergSaxton'   # Gerchberg-Saxton Algorithm
    HIO     =   'HIO_Fineup'        # Hybrid Input-Output Fineup's Algorithm
    W_ROI   =   'Weighted_ROI'      # Weighted Region of Interest
    Wang    =   'Wang'              # Wang's Algorithm
    MRAF    =   'MRAF'              # Mixed Region Amplitude Freedom

#
class MetricType:
    Uniformity  =   'Uniformity'   
    MeanError   =   'MeanError'  
    MaxError    =   'MaxError'
 
class IlluminPatterns:

    def Gaussian(self,(height,width),miu,sigma):
        x,y = np.meshgrid(range(height),range(width))
        d=np.sqrt(x*x+y*y)
        return 255*np.exp(-(d-miu)**2/(2.0*sigma**2))

class SLM_Phase:
    
    def __init__(self,illum=1,img=None,type=None):
        if img.any() != None:
             self.SetPattern(img)
        if type != None:
            self.SetAlgorithm(type)
        self.uniformity=[]

    def SetIllumination(self,illum):
        self.illum=illum

    # Set the desired pattern to achieve at the objective focal plane
    def SetPattern(self,img):
        self.target_amp=np.sqrt(img)
        self.ROI=self.target_amp>0
        self.height,self.width=img.shape
        self.phase_slm=np.random.rand(self.height,self.width)
        self.phase_obj=np.random.rand(self.height,self.width)
    
    # Choose the algorithm to be used
    def SetAlgorithm(self,type):
        self.__Algorithm = AlgorithmCreator().FactoryMethod(type,self)
    
    # Itterate Phase Retrieval algorithm
    def Compute(self,num_iter=30,eps=None):
        self.__Algorithm.Compute()

    def ConstructedPattern(self):
        recover = np.fft.fft2(np.exp(1j*self.phase_slm))
        recover = np.fft.fftshift(recover)
        recover = np.abs(recover)**2
        return recover

    def appendUniformity(self, U_o):
        mini=np.min(np.abs(U_o[self.ROI]))
        maxi=np.max(np.abs(U_o[self.ROI]))
        uniform=1-(maxi-mini)/(maxi+mini)
        self.uniformity.append(uniform)

    def Plot(self,frame=None):

        focus_region=np.full(self.target_amp.shape,True)
        if frame!=None:
            focus_region=np.ix_(range(frame[0],frame[1]),range(frame[2],frame[3]))

        plt.figure(1)
        plt.subplot(131)
        plt.imshow(self.target_amp[focus_region])
        plt.title('Zoomed desired')
        plt.subplot(132)
        plt.imshow(self.phase_slm)
        plt.title('Phase mask')
        plt.subplot(133)
        
        recovered   =   self.ConstructedPattern()

        plt.imshow(recovered[focus_region])
        plt.title('Zoomed recovered')
    


class AlgorithmCreator:
    def FactoryMethod(self,type,SLM_handler):
        return globals()[type](SLM_handler)
        

class PhaseRetrieval:
    def __init__(self,SLM_handler):
        self.SLM_handler=SLM_handler
        self.A_x=SLM_handler.target_amp
        self.ROI=SLM_handler.target_amp>0

    def Objective(self,U_o):         # Transformation from U_o to U_x
        phase_o=np.angle(U_o)
        U_x=(self.A_x)*np.exp(1j*phase_o)
        return U_x
    
    def SLM(self,g_s):                  # Transformation from desired modulation to SLM modulation(U_s)
        return np.exp(1j*np.angle(g_s))

    def SLM_To_Objective(self,U_s):
        U_o=np.fft.fft2(U_s)
        U_o=np.fft.fftshift(U_o)
        self.SLM_handler.appendUniformity(U_o)
        return U_o

    def Objective_To_SLM(self,U_x):
        g_s=np.fft.ifftshift(U_x)
        g_s=np.fft.ifft2(g_s)
        return g_s

    def Compute(self):
        g_s=self.SLM_handler.illum*np.exp(1j*self.SLM_handler.phase_slm)
        for num_iter in range(30):
            U_s=self.SLM(g_s)
            self.SLM_handler.phase_slm=np.angle(U_s)
            U_o=self.SLM_To_Objective(U_s)
            U_x=self.Objective(U_o)
            g_s=self.Objective_To_SLM(U_x)
    
            


class GerchbergSaxton(PhaseRetrieval):
    pass

class Weighted_ROI(PhaseRetrieval):

    def Objective(self,U_o):         # Transformation from U_o to U_x
        try:
            eps=np.linalg.norm(self.__last_obj-U_o)
        except:
            self.__last_obj=np.empty_like(U_o)
            eps=np.linalg.norm(U_o)
        phase_o=np.angle(U_o)
        self.A_x[self.ROI]=np.multiply(np.divide(self.normalization(self.SLM_handler.target_amp[self.ROI]),self.normalization(np.abs(U_o[self.ROI]))),self.A_x[self.ROI])
        plt.imshow(np.abs(U_o)[500-50:500+50,500-50:500+50]/np.linalg.norm(U_o))
        plt.show()
        U_x=self.A_x*np.exp(1j*phase_o)
        return U_x
    
    def normalization(self,input):
        mini=np.min(input)
        maxi=np.max(input)
        input[input<(maxi+mini)/2]=mini+0.8*(maxi-mini)
        return input/np.linalg.norm(input)

class MRAF(PhaseRetrieval):

    def Objective(self,U_o):
        try:
            eps=np.linalg.norm(self.__last_obj-U_o)
        except:
            self.__last_obj=np.empty_like(U_o)
            eps=np.linalg.norm(U_o)
        phase_o=np.angle(U_o)
        M=self.SLM_handler.uniformity[-1]
        self.A_x=((1+M)*self.norm(self.SLM_handler.target_amp)-M*self.norm(np.abs(U_o)))*self.A_x
        U_x=self.A_x*np.exp(1j*phase_o)
        return U_x

    def norm(self,input):
        return input/np.linalg.norm(input)

'''

class GerchbergSaxton:

    def __init__(self,img):
        self.target=np.sqrt(img/255)
        self.height,self.width=img.shape
        self.slm_phase=np.random.rand(self.height,self.width)
        self.uniformity=[]
        

    def compute(self,n_iter):
        obj=np.empty_like(self.target,dtype="complex")
        slm=np.empty_like(self.target,dtype="complex")
        self.w=self.target
        for num in range(n_iter):
            slm=np.exp(1j*self.slm_phase)

            obj=np.fft.fft2(slm)
            obj=np.fft.fftshift(obj)

            obj_int=np.abs(obj)**2
            norm_i=self.normalization(obj_int)

            phase=np.angle(obj)

            self.feed(norm_i)

            obj=self.w*np.exp(1j*phase)

            obj=np.fft.ifftshift(obj)
            slm=np.fft.ifft2(obj)

            self.slm_phase=np.angle(slm)

    def normalization(self,intensity):
        maxi=np.max(intensity)
        mini=np.min(intensity)
        return (intensity-mini)/(maxi-mini)

    def feed(self,norm_i):
        self.check_uniformity(norm_i)
        self.w=self.target

    def check_uniformity(self,x_int):
        maxi=np.max(x_int[self.target>0.75])
        mini=np.min(x_int[self.target>0.75])
        uniformity=1-(maxi-mini)/(maxi+mini)
        self.uniformity.append(uniformity)
        return uniformity


class MRAF(GerchbergSaxton):
    def feed(self,norm_i):
        corr_zone=self.target>0.75
        M=self.check_uniformity(norm_i)
        self.w[corr_zone]=self.target[corr_zone]+M*(self.target[corr_zone]-np.sqrt(norm_i[corr_zone]))
        self.w=normalization(self.w)

class WGS(GerchbergSaxton):
    def feed(self,norm_i):
        corr_zone=self.target>0.75
        self.w[corr_zone]=np.sqrt(self.target[corr_zone]/np.sqrt(norm_i[corr_zone]))*self.w[corr_zone]
        self.w=self.normalization(self.w)
        self.check_uniformity(norm_i)

'''