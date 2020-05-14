#include "display.h"

ImagePR::ImagePR(int height, int width): width(width),height(height)
{
    image=cv::Mat(height,width,CV_8UC3);
    gray_image=cv::Mat(height,width,CV_8U);
    gray_array=(float*)malloc(height*width*sizeof(float));
}

ImagePR::ImagePR(const char *path){
    image = cv::imread( path, CV_8S);
    if( !image.data )
	{
	  printf( "No image data \n" );
	  exit(-1);
	}
    width = image.cols;
    height = image.rows;
    cv::cvtColor(image,gray_image,cv::COLOR_BGR2GRAY);
    gray_array=(float*)malloc(height*width*sizeof(float));
}

ImagePR::~ImagePR()
{
    free(gray_array);
    printf("Image object destructed successfully!\n");
}

float* ImagePR::GetGray(){
    float temp;
    for(int i=0;i<height;i++)
        for(int j=0;j<width;j++){
            //cv::Vec3b color = image.at<cv::Vec3b>(cv::Point(j,i));
            //gray_array[i*width+j]=float(0.3*color[0]+0.3*color[1]+0.3*color[2]);
            unsigned char color = gray_image.at<unsigned char>(cv::Point(j,i));
            gray_array[i*width+j]=(float)color;
        }
    return gray_array;
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

void ImagePR::show(const char *title){
    cv::namedWindow(title, cv::WINDOW_AUTOSIZE );
    cv::imshow(title, gray_image);
}