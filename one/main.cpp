#include <iostream> 
#include <opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;

#define _src "C:\\Users\\Cashion\\Desktop\\something\\lena.jpg"    //原图像路径
#define _dst1_ "C:\\Users\\Cashion\\Desktop\\something\\object1.jpg"    //生成图像路径1
#define _dst2_ "C:\\Users\\Cashion\\Desktop\\something\\object2.jpg"    //生成图像路径2
#define _dst3_ "C:\\Users\\Cashion\\Desktop\\something\\object3.jpg"    //生成图像路径3


Mat addSaltNoise(const Mat srcImage, int n);   //椒盐化噪声函数声明
int main()
{
	Mat inputim;
	Mat im;                                 //读取原图
	im = imread(_src,0);         
	namedWindow("原图", 1);
	imshow("原图", im);                    //原图为im

	inputim = im;
	Mat im1 = addSaltNoise(inputim, 8000);    //加椒盐噪声处理
	namedWindow("加噪", 1);
	imshow("加噪", im1);
	imwrite(_dst1_,im1);                       //输出为im1

	inputim = im;
	Mat im2;                               //平滑（均值）滤波处理
	blur(inputim, im2, Size(3,3));
	namedWindow("平滑滤波", 1);
	imshow("平滑滤波", im2);
	imwrite(_dst2_, im2);                  //输出位im2


	inputim = im1;
	Mat im3;                               //中值滤波处理
	medianBlur(inputim, im3, 3);
	namedWindow("中值滤波", 1);
	imshow("中值滤波", im3);
	imwrite(_dst3_, im3);                 //输出为Im3

	waitKey(0);
	return 0;
}



