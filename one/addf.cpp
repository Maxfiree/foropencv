#include <iostream> 
#include <opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;

/*********************************************************************
*                           椒盐化噪声函数                           *
*********************************************************************/
Mat addSaltNoise(const Mat srcImage, int n)    
{
	Mat dstImage = srcImage.clone();
	srand(time(NULL));
	for (int k = 0; k < n; k++)
	{
		//随机取值行列  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;       //盐噪声  
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;     //椒噪声  
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
};
