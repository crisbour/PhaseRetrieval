import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import csv
import os
import numpy as np

class dataVect:
    def __init__(self):
        self.data=[]
        self.name=''
    def SetName(self,name_recieved):
        self.name=name_recieved
    def Name(self):
        return self.name
    def append(self,value):
            self.data.append(float(value))
    def getData(self):
        return self.data

def main():
    data=[]
    index=0
    for file in os.listdir("."):
        if file.endswith(".txt"):
            data.append(dataVect())
            data[index].SetName(file[:-4])
            print(file)
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count=0
                for row in csv_reader:
                    if line_count==0:
                        print(f'Column names are {". ".join(row)}')
                    else:
                        data[index].append(row[0])
                    line_count +=1
                print(f'Processed {line_count} lines.')
            index+=1

    
    for i in range(index):
        figU=plt.figure().add_subplot(1,1,1)
        figU.set_xlabel('Pixel index')
        figU.set_ylabel('Irradiance')
        figU.plot(data[i].getData(),label=data[i].Name())
        figU.legend()
            
    plt.show()
    

if __name__=="__main__":
    main()
