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

#include "algorithm.h"

#define N 1024
#define M 1024

int main(int argc, char **argv){
	cv::Mat image;

	image = cv::imread( argv[1], 1 );
	if( argc != 2 || !image.data )
	{
	  printf( "No image data \n" );
	  return -1;
	}

	PhaseRetrieve transfs(N,M,Gerchberg_Saxton);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	transfs.Test();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	printf("Elapse time: %f milliseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0);

	cv::namedWindow( "Display Image", cv::WINDOW_AUTOSIZE );
	imshow( "Display Image", image );
	cv::waitKey(0);
	return 0;
}


