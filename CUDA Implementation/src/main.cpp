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


int main(int argc, char **argv){

	ImagePR desired(1000,1000); 
	// Square square1(desired,10,10);
	// Pattern pattern1(desired,3,3,30,30);
	// pattern1.setElement(square1);
	// pattern1.Draw(115,115);
	MeshPattern pattern(desired,2,150,new Square(desired,60,60));
	pattern.Draw(0,0);

	ImagePR illumination(desired.GetHeight(),desired.GetWidth(),cv::COLORMAP_JET);
	illumination.MakeGaussian((desired.GetWidth()-1)/2.0,(desired.GetHeight()-1)/2.0,5000000.,5000000.);

	PhaseRetrieve transfs(desired.GetGray(),desired.GetHeight(),desired.GetWidth(),Gerchberg_Saxton);
	transfs.SetIllumination(illumination.GetGray());

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	transfs.Test();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	printf("Elapse time: %f milliseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0);

	ImagePR reconst(desired.GetHeight(),desired.GetWidth());
	reconst.SetGray(transfs.GetImage());

	ImagePR phase(desired.GetHeight(),desired.GetWidth(),cv::COLORMAP_HOT);
	phase.SetGray(transfs.GetPhaseMask());

	illumination.show("Illumination");
	desired.show("Desired Image");
	phase.show("Phase Mask");
	reconst.show("Reconstructed Image");
	// cv::Mat image(desired.GetHeight(),desired.GetWidth(),CV_8U);
    // cv::putText(image, "Hello World!", cv::Point( 100, 200 ), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 1.0, cv::Scalar( 0, 0, 0 ));
    // cv::imshow("My Window", image);
	cv::waitKey(0);
	return 0;
}


