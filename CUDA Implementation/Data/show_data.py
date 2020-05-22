import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import csv
import os
import numpy as np

class dataVect:
    def __init__(self):
        self.data=[]
        self.fields=[]
        self.name=''
    def SetName(self,name_recieved):
        self.name=name_recieved
    def NameFields(self,*argv):
        for names in argv:
            self.fields.append(names)
            self.data.append([])
    def Name(self):
        return self.name
    def append(self,*argv):
        index=0
        for value_of_field in argv:
            self.data[index].append(float(value_of_field))   
            index += 1
    def getData(self,index):
        return self.data[index]
    def getFieldName(self,index):
        return self.fields[index]

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
                    elif line_count==1:
                        print(index)
                        data[index].NameFields(row[0],row[1],row[2])
                    else:
                        data[index].append(row[0],row[1],row[2])
                    line_count +=1
                print(f'Processed {line_count} lines.')
            index+=1
            
    figU=plt.figure().add_subplot(1,1,1)
    figA=plt.figure().add_subplot(1,1,1)
    figE=plt.figure().add_subplot(1,1,1)
    
    figU.set_xlabel('Iteration')
    figA.set_xlabel('Iteration')
    figE.set_xlabel('Iteration')
    
    figU.set_ylabel(data[0].getFieldName(0))
    figA.set_ylabel(data[0].getFieldName(1))
    figE.set_ylabel(data[0].getFieldName(2))
    
    for i in range(index):
            figU.plot(data[i].getData(0),label=data[i].Name())
            figA.plot(data[i].getData(1),label=data[i].Name())
            figE.plot(data[i].getData(2),label=data[i].Name())
            
    figU.legend()
    figA.legend()
    figE.legend()
            
    plt.show()
    

if __name__=="__main__":
    main()
