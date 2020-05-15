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
    void SetPixel(int x, int y);
    void show(const char *title);
};

class Drawing{
public:
    virtual void Draw(int x, int y)=0;
};

class Square:public Drawing{
private:
    int dx,dy;
    ImagePR &image;
public:
    Square(ImagePR &image,int dx, int dy):image(image),dx(dx),dy(dy){};
    ~Square(){};
    void Draw(int x, int y){
        for(int i=0;i<dx;i++)
            for(int j=0;j<=dy;j++)
            image.SetPixel(x+i,y+j);
    };
};
class Pattern:public Drawing{
private:
    int dx,dy,nx,ny;
    ImagePR &image;
    Drawing *pattern_elem;
public:
    Pattern(ImagePR &image,int nx,int ny, int dx, int dy):image(image),nx(nx),ny(ny),dx(dx),dy(dy){};
    ~Pattern(){};
    void setElement(Drawing &elem){pattern_elem=&elem;};
    void Draw(int x, int y){
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++)
                pattern_elem->Draw(x+i*dx,y+j*dy);
    };
};

#endif