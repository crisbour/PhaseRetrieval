/*
 * display.h
 *
 *  Created on: 10 May 2020
 *      Author: Cristian Bourceanu
 */
#ifndef __IMAGEPR_H__
#define __IMAGEPR_H__

#include <opencv2/opencv.hpp>
#include <cmath>


class ImagePR
{
protected:
    float *gray_array;
    int width, height;
    cv::Mat image;
    cv::Mat gray_image;
    int color_display;
public:
    ImagePR(int height, int width,int coloring=cv::COLORMAP_HOT);
    ImagePR(const char *path,int coloring=cv::COLORMAP_HOT);
    ~ImagePR();
    float* GetGray();
    void SetGray(float *array);
    void MakeGaussian(float x_0,float y_0,float var_x, float var_y);
    int GetWidth()const{return width;};
    int GetHeight()const{return height;};
    void show(const char *title);
};


#endif