#include <iostream> 
#include <opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;

/*********************************************************************
*                           ���λ���������                           *
*********************************************************************/
Mat addSaltNoise(const Mat srcImage, int n)    
{
	Mat dstImage = srcImage.clone();
	srand(time(NULL));
	for (int k = 0; k < n; k++)
	{
		//���ȡֵ����  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;       //������  
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
		//���ȡֵ����  
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//ͼ��ͨ���ж�  
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;     //������  
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
