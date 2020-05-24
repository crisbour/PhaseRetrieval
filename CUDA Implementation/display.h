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

class ImagePR_Interface{
protected:
    int width,height;
public:
    virtual ~ImagePR_Interface(){};
    virtual void SetDimensions(int _width, int _height){width=_width;height=_height;};
    virtual float* GetGray()=0;
    virtual void SetGray(float *array)=0;
    virtual void MakeGaussian(float x_0,float y_0,float var_x, float var_y)=0;
    virtual int GetWidth()const{return width;};
    virtual int GetHeight()const{return height;};
    virtual void SetPixel(int x, int y, int value=1)=0;
    virtual void ErasePixel(int x, int y, int value=1){SetPixel(x,y,0);};
    virtual unsigned char GetPixel(int x, int y)=0;
    virtual void show(const char *title)=0;
};
class ImagePR:public ImagePR_Interface
{
protected:
    float *gray_array=NULL;
    cv::Mat image;
    cv::Mat gray_image;
    int color_display;
public:
    ImagePR(int width, int height,int coloring=cv::COLORMAP_HOT);
    ImagePR(const char *path,int coloring=cv::COLORMAP_HOT);
    ~ImagePR();
    float* GetGray();
    void SetGray(float *array);
    void SetGray(float *array, int width_original, int height_original);
    void MakeGaussian(float x_0,float y_0,float var_x, float var_y);
    int GetWidth()const{return width;};
    int GetHeight()const{return height;};
    void SetPixel(int x, int y, int value=1);
    unsigned char GetPixel(int x, int y);
    void show(const char *title);
};

class Drawing{
public:
    ~Drawing(){};
    virtual int Width()=0;
    virtual int Height()=0;
    virtual void Draw(int x, int y)=0;
};

class Circle:public Drawing{
protected:
    ImagePR_Interface &image;
    float radius;
public:
    Circle(ImagePR_Interface &image, int radius):image(image),radius(radius){};
    ~Circle(){};
    int Width(){return 2* (int)radius;}
    int Height(){return 2* (int)radius;}
    void Draw(int x, int y);
};

class Square:public Drawing{
protected:
    ImagePR_Interface &image;
    int dx,dy;
public:
    Square(ImagePR_Interface &image,int dx, int dy):image(image),dx(dx),dy(dy){};
    int Width(){return dx;};
    int Height(){return dy;};
    ~Square(){};
    void Draw(int x, int y){
        for(int i=0;i<dx;i++)
            for(int j=0;j<dy;j++)
            image.SetPixel(x+i,y+j);
    };
};
class Pattern:public Drawing{
private:
    ImagePR_Interface &image;
    int dx,dy,nx,ny;
    Drawing &elem;
public:
    Pattern(ImagePR_Interface &_image,int _nx,int _ny, int _dx, int _dy,Drawing &elem):image(image),nx(_nx),ny(_ny),dx(_dx),dy(_dy),elem(elem){};
    virtual ~Pattern(){};
    int Width(){return dx*(nx-1)+nx*elem.Width();};
    int Height(){return dx*(ny-1)+ny*elem.Height();};
    //void setElement(Drawing &_elem){&elem=_elem;};
    void Draw(int x, int y){
        int Tx=dx+elem.Width();
        int Ty=dy+elem.Height();
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++)
                elem.Draw(x+i*Tx,y+j*Ty);
    };
};
class MeshPattern:public Drawing{
private:
    ImagePR_Interface &image;
    int n, step;
    Drawing &elem;
public:
    MeshPattern(ImagePR_Interface &image,int n, int step, Drawing &_elem):image(image),elem(_elem),n(n),step(step){};
    ~MeshPattern(){};
    int Width(){return step*(n-1)+n*elem.Width();};
    int Height(){return step*(n-1)+n*elem.Height();};
    void Draw(int x, int y){
        Pattern* pattern=new Pattern(image,n,n,step,step,elem);
        int wi=image.GetWidth(); int hi=image.GetHeight();
        int wd=pattern->Width(); int hd=pattern->Height();
        pattern->Draw(x+(wi-wd)/2,y+(hi-hd)/2);
        delete pattern;
    };
};

#endif