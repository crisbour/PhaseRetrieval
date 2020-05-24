/*
 * main.cpp
 *
 *  Created on: 9 May 2020
 *      Author: Cristian Bourceanu
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "display.h"
#include "algorithm.h"

void printMetrics(const char* data_name,std::vector<std::vector<float>>data){
	FILE *File;
	char filePath[100]="../Data/";
	strcat(filePath,data_name); strcat(filePath,".txt");
	printf("Print data to %s\n",filePath);
	File=fopen(filePath,"w+");
	fprintf(File,"%u\n",data[0].size());
	fprintf(File,"Uniformity,RMS Error,Efficiency\n");
	for(unsigned int i=0;i<data[0].size();i++){
		fprintf(File,"%f,%f,%f\n",data[0][i],data[1][i],data[2][i]);
	}

}

void printSlice(PhaseRetrieve &reconst){
	const char *data_name=reconst.GetName();
	float *gray_image=reconst.GetImage();
	int nx=500,ny=500;
	FILE *File;
	char filePath[100]="../Data/Slice/";
	strcat(filePath,data_name); strcat(filePath,".txt");
	printf("Print data to %s\n",filePath);
	File=fopen(filePath,"w+");
	fprintf(File,"%u\n",nx);
	for(int i=0;i<nx;i++)
		fprintf(File,"%f\n",gray_image[i+nx*(ny/2)]/255.0);
}
int main(int argc, char **argv){
	// Square square1(desired,10,10);
	// Pattern pattern1(desired,3,3,30,30);
	// pattern1.setElement(square1);
	// pattern1.Draw(115,115);
	int spacing=10;
	int nx_ny=10;
	ImagePR_Interface *desired;
	if(argc==2)							//Read image
		desired= new ImagePR(argv[1]);
	else if(argc==3){					//Create ring with outter radius argv[1] amd inner radius arv[2]
		desired= new ImagePR(500,500);
		Circle circle(*desired,atoi(argv[1]));
		Circle erase_circle(*desired,atoi(argv[2]));
		circle.Draw(-1,-1);
		erase_circle.Draw(-1,-1);
	}
	else{								//Create multi foci points mesh
		desired= new ImagePR(500,500); 
		Square spot(*desired,10,10);
		MeshPattern pattern(*desired,nx_ny,spacing,spot);
		pattern.Draw(0,0);
	}

	ImagePR illumination(desired->GetHeight(),desired->GetWidth(),cv::COLORMAP_JET);
	illumination.MakeGaussian((desired->GetWidth()-1)/2.0,(desired->GetHeight()-1)/2.0,10000.,10000.);

	printf("(Width,Height)=(%d,%d)\n",desired->GetWidth(),desired->GetHeight());
	PhaseRetrieve transfs(desired->GetGray(),desired->GetWidth(),desired->GetHeight(),UCMRAF);
	transfs.SetIllumination(illumination.GetGray());
	
	//transfs.SetROI(249.5,249.5,150);		//If ROI is not set specifically, SR will be used by default
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// transfs.Compute(1);
	// transfs.SetAlgorithm(Weighted_GS);
	transfs.Compute(50);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	printf("Elapse time: %f milliseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0);

	printf("%s\n",transfs.GetName());
	printMetrics(transfs.GetName(),transfs.GetMetrics());
	printSlice(transfs);

	ImagePR ROI_mask(desired->GetWidth(),desired->GetHeight());
	ROI_mask.SetGray(transfs.GetROIMask());

	//ImagePR reconst(floor(nx_ny*spacing*1.4),floor(nx_ny*spacing*1.4));
	ImagePR reconst(desired->GetWidth(),desired->GetHeight());
	reconst.SetGray(transfs.GetImage(),desired->GetWidth(),desired->GetHeight());

	ImagePR phase(desired->GetHeight(),desired->GetWidth(),cv::COLORMAP_TWILIGHT_SHIFTED);
	phase.SetGray(transfs.GetPhaseMask());

	illumination.show("Illumination");
	desired->show("Desired Image");
	ROI_mask.show("ROI Mask");
	phase.show("Phase Mask");
	reconst.show("Reconstructed Image");

	cv::waitKey(0);
	delete desired;
	return 0;
}


