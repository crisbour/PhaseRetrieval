#ifndef __IMAGEPR_H__
#define __IMAGEPR_H__

#include <opencv2/opencv.hpp>


class ImagePR
{
protected:
    float *gray_array;
    int width, height;
    cv::Mat image;
    cv::Mat gray_image;
public:
    ImagePR(int height, int width);
    ImagePR(const char *path);
    ~ImagePR();
    float* GetGray();
    void SetGray(float *array);
    int GetWidth()const{return width;};
    int GetHeight()const{return height;};
    void show(const char *title);
};


#endif