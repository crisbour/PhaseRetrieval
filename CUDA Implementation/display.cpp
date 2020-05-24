/*
 * display.cpp
 *
 *  Created on: 10 May 2020
 *      Author: Cristian Bourceanu
 */
#include "display.h"

void space2underscore(char *str,int length){
    for(int i=0;i<length;i++)
        if(str[i]==' ')
            str[i]='_';
}

ImagePR::ImagePR(int width, int height,int coloring):color_display(coloring)
{
    SetDimensions(width,height);
    image=cv::Mat(height,width,CV_8UC3);
    gray_image=cv::Mat(height,width,cv::IMREAD_GRAYSCALE);
    printf("Dim=%d:\n",height*width);
    gray_array=(float*)malloc(height*width*sizeof(float));
    printf("Memory allocated for blank image\n");
}

ImagePR::ImagePR(const char *path,int coloring):color_display(coloring){
    image = cv::imread( path, cv::IMREAD_COLOR);
    if( !image.data )
	{
	  printf( "No image data \n" );
	  exit(-1);
	}
    SetDimensions(image.cols,image.rows);

    cv::cvtColor(image,gray_image,cv::COLOR_BGR2GRAY);
    gray_array=(float*)malloc(height*width*sizeof(float));
    printf("Memory allocated for inserted image\n");
}

ImagePR::~ImagePR()
{
    if(gray_array)
        free(gray_array);
    printf("Image object destructed successfully!\n");
}

float* ImagePR::GetGray(){
     printf("Get %d\n",height*width);
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++){
            //cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(j,i));
            //gray_array[i*width+j]=float(0.3*color[0]+0.3*color[1]+0.3*color[2]);
            unsigned char color = gray_image.at<unsigned char>(cv::Point(j,i));
            gray_array[i*width+j]=(float)color;
        }
    return gray_array;
}

void ImagePR::SetGray(float *array, int width_original, int height_original){
    int y_o,x_o;
    for(int y=0;y<height;y++)
        for(int x=0;x<width;x++){
            //image.at<cv::Vec3b>(x,y)[0] = 200;
            y_o=y+(height_original-height)/2;
            x_o=x+(width_original-width)/2;
            cv::Vec3b & color = image.at<cv::Vec3b>(cv::Point(x,y));
            unsigned char & colorG = gray_image.at<unsigned char>(cv::Point(x,y));
            colorG=color[2]=color[1]=color[0]=(unsigned char) array[y_o*width_original+x_o];
            gray_array[y*width+x]=array[y_o*width_original+x_o];
        }
}
void ImagePR::SetGray(float *array){
    for(int y=0;y<height;y++)
        for(int x=0;x<width;x++){
            //image.at<cv::Vec3b>(x,y)[0] = 200;
            cv::Vec3b & color = image.at<cv::Vec3b>(cv::Point(x,y));
            unsigned char & colorG = gray_image.at<unsigned char>(cv::Point(x,y));
            colorG=color[2]=color[1]=color[0]=(unsigned char) array[y*width+x];
            gray_array[y*width+x]=array[y*width+x];
        }
}

void ImagePR::MakeGaussian(float x_0,float y_0,float var_x, float var_y){
    float val=0;
    for(int y=0;y<height;y++)
        for(int x=0;x<width;x++){
            val=255*exp(-(pow(x-x_0,2)/(2*var_x)+pow(y-y_0,2)/(2*var_y)));
            cv::Vec3b & color = image.at<cv::Vec3b>(cv::Point(x,y));
            unsigned char & colorG = gray_image.at<unsigned char>(cv::Point(x,y));
            colorG=color[2]=color[1]=color[0]=(unsigned char) val;
            gray_array[y*width+x]=val;
        }
}
void ImagePR::SetPixel(int x, int y, int value){
    cv::Vec3b & color = image.at<cv::Vec3b>(cv::Point(x,y));
    unsigned char & colorG = gray_image.at<unsigned char>(cv::Point(x,y));
    colorG=color[2]=color[1]=color[0]=255*value;
    gray_array[y*width+x]=255*value;
}

unsigned char ImagePR::GetPixel(int x, int y){
    return (unsigned char)gray_array[y*width+x];
}

void ImagePR::show(const char *title){
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE );
    cv::Mat im_color;
    cv::applyColorMap(gray_image,im_color,color_display);
    char buffer[100]="../Images/";    strcat(buffer,title);   strcat(buffer,".png");  space2underscore(buffer,100);
    printf("%s writing path is: %s\n",title,buffer);
    cv::imwrite(buffer,im_color);
    cv::imshow(title, im_color);
}







// *****************  DRAWING *********************//

void Circle::Draw(int x, int y){
	char *not_checked;
    float x_C,y_C;
    int nx=image.GetWidth();
    int ny=image.GetHeight();
	not_checked=new char[nx*ny];
	for(int i=0;i<nx*ny;i++)
		not_checked[i]=1;
	std::queue<int> queuex;
	std::queue<int> queuey;
    if(x<0 || y_C<0){
        x_C=(image.GetWidth()-1)/2.0;
        y_C=(image.GetHeight()-1)/2.0;
    }
    else
    {
        x_C=x; y_C=y;
    }
	int x_pix=floor(x_C); int y_pix=floor(y_C);

	if(pow(x_pix-x_C,2)+pow(y_pix-y_C,2)<=radius*radius){
		not_checked[x_pix+y_pix*nx]=0;
		queuex.push(x_pix);
		queuey.push(y_pix);
	}
	int posx[]={-1,0,1,-1,1,-1,0,1};
	int posy[]={-1,-1,-1,0,0,1,1,1};
	while(!queuex.empty()&&!queuey.empty()){
		x_pix=queuex.front(); 
		y_pix=queuey.front(); 
		queuex.pop();
		queuey.pop();
        if(image.GetPixel(x_pix,y_pix))
            image.ErasePixel(x_pix,y_pix);
        else
            image.SetPixel(x_pix,y_pix);
		for(int i=0;i<8;i++)
			if(pow(x_pix+posx[i]-x_C,2)+pow(y_pix+posy[i]-y_C,2)<=radius*radius && not_checked[x_pix+posx[i]+(y_pix+posy[i])*nx]){
				not_checked[x_pix+posx[i]+(y_pix+posy[i])*nx]=0;
				queuex.push(x_pix+posx[i]);
				queuey.push(y_pix+posy[i]);
			}
				
	}
	delete[] not_checked;
}