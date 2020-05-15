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
    int Width(){return 0;}
    int Height(){return 0;}
};

class Square:public Drawing{
private:
    int dx,dy;
    ImagePR &image;
public:
    Square(ImagePR &image,int dx, int dy):image(image),dx(dx),dy(dy){};
    int Width()const{return dx;};
    int Height()const{return dx;};
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
    Drawing *elem;
public:
    Pattern(ImagePR &image,int nx,int ny, int dx, int dy):image(image),nx(nx),ny(ny),dx(dx),dy(dy){};
    ~Pattern(){};
    int Width()const{return dx*(nx-1)+elem->Width();};
    int Height()const{return dx*(nx-1)+elem->Height();};
    void setElement(Drawing &_elem){elem=&_elem;};
    void Draw(int x, int y){
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++)
                elem->Draw(x+i*dx,y+j*dy);
    };
};
class MeshPattern:public Drawing{
private:
    int step,n;
    Drawing &elem;
    ImagePR &image;
public:
    MeshPattern(ImagePR &image,int n, int step, Drawing *_elem):image(image),elem(*_elem),n(n),step(step){};
    ~MeshPattern(){delete &elem;};
    void Draw(int x, int y){
        Pattern* pattern=new Pattern(image,n,n,step,step);
        pattern->setElement(elem);
        int wi=image.GetWidth(); int hi=image.GetHeight();
        int wd=pattern->Width(); int hd=pattern->Height();
        pattern->Draw(x+(wi-wd)/2,y+(hi-hd)/2);
        delete pattern;
    };
};

#endif