/*
 * main.cu
 *
 *  Created on: 11 May 2020
 *      Author: cristi
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "display.h"
#include "algorithm.h"


int main(int argc, char **argv){

	ImagePR desired(argv[1]); 

	PhaseRetrieve transfs(desired.GetGray(),desired.GetHeight(),desired.GetWidth(),Gerchberg_Saxton);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	transfs.Test();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	printf("Elapse time: %f milliseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0);

	ImagePR reconst(desired.GetHeight(),desired.GetWidth());
	reconst.SetGray(transfs.GetImage());

	ImagePR phase(desired.GetHeight(),desired.GetWidth());
	phase.SetGray(transfs.GetPhaseMask());

	desired.show("Desired Image");
	reconst.show("Reconstructed Image");
	phase.show("Phase Mask");
	// cv::Mat image(desired.GetHeight(),desired.GetWidth(),CV_8U);
    // cv::putText(image, "Hello World!", cv::Point( 100, 200 ), cv::FONT_HERSHEY_SIMPLEX | cv::FONT_ITALIC, 1.0, cv::Scalar( 0, 0, 0 ));
    // cv::imshow("My Window", image);
	cv::waitKey(0);
	return 0;
}


