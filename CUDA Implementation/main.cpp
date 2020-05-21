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

void printData(const char* data_name,std::vector<std::vector<float>>data){
	FILE *File;
	char filePath[100]="../Data/";
	strcat(filePath,data_name); strcat(filePath,".txt");
	File=fopen(filePath,"w+");
	fprintf(File,"%u\n",data[0].size());
	fprintf(File,"Uniformity Accuracy Efficiency\n");
	for(unsigned int i=0;i<data[0].size();i++){
		fprintf(File,"%f %f %f\n",data[0][i],data[1][i],data[2][i]);
	}

}
int main(int argc, char **argv){
	// Square square1(desired,10,10);
	// Pattern pattern1(desired,3,3,30,30);
	// pattern1.setElement(square1);
	// pattern1.Draw(115,115);
	int spacing=12;
	int nx_ny=5;
	ImagePR_Interface *desired;
	if(argc==2)
		desired= new ImagePR(argv[1]);
	else{
		desired= new ImagePR(200,200); 
		Square spot(*desired,5,5);
		MeshPattern pattern(*desired,nx_ny,spacing,spot);
		pattern.Draw(0,0);
	}

	ImagePR illumination(desired->GetHeight(),desired->GetWidth(),cv::COLORMAP_JET);
	illumination.MakeGaussian((desired->GetWidth()-1)/2.0,(desired->GetHeight()-1)/2.0,10000.,10000.);

	printf("(Width,Height)=(%d,%d)\n",desired->GetWidth(),desired->GetHeight());
	PhaseRetrieve transfs(desired->GetGray(),desired->GetWidth(),desired->GetHeight(),Gerchberg_Saxton);
	transfs.SetIllumination(illumination.GetGray());
	
	//transfs.SetROI(959.5,539.5,100);		\\If ROI is not set specifically, SR will be used by default
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	transfs.Compute(50);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	printf("Elapse time: %f milliseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0);

	printf("%s\n",transfs.GetName());
	printData(transfs.GetName(),transfs.GetMetrics());


	//ImagePR reconst(floor(nx_ny*spacing*1.4),floor(nx_ny*spacing*1.4));
	ImagePR reconst(desired->GetWidth(),desired->GetHeight());
	reconst.SetGray(transfs.GetImage(),desired->GetWidth(),desired->GetHeight());

	ImagePR phase(desired->GetHeight(),desired->GetWidth(),cv::COLORMAP_TWILIGHT_SHIFTED);
	phase.SetGray(transfs.GetPhaseMask());

	illumination.show("Illumination");
	desired->show("Desired Image");
	phase.show("Phase Mask");
	reconst.show("Reconstructed Image");
	// cv::Mat image(desired.GetHeight(),desired.GetWidth(),CV_8U);
    // cv::putText(image, "Hello World!", cv::Point( 100, 200 ), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 1.0, cv::Scalar( 0, 0, 0 ));
    // cv::imshow("My Window", image);
	cv::waitKey(0);
	delete desired;
	return 0;
}


