#include <iostream> 
#include <opencv.hpp>
#include <time.h>
using namespace std;
using namespace cv;

#define _src "C:\\Users\\Cashion\\Desktop\\something\\lena.jpg"    //ԭͼ��·��
#define _dst1_ "C:\\Users\\Cashion\\Desktop\\something\\object1.jpg"    //����ͼ��·��1
#define _dst2_ "C:\\Users\\Cashion\\Desktop\\something\\object2.jpg"    //����ͼ��·��2
#define _dst3_ "C:\\Users\\Cashion\\Desktop\\something\\object3.jpg"    //����ͼ��·��3


Mat addSaltNoise(const Mat srcImage, int n);   //���λ�������������
int main()
{
	Mat inputim;
	Mat im;                                 //��ȡԭͼ
	im = imread(_src,0);         
	namedWindow("ԭͼ", 1);
	imshow("ԭͼ", im);                    //ԭͼΪim

	inputim = im;
	Mat im1 = addSaltNoise(inputim, 8000);    //�ӽ�����������
	namedWindow("����", 1);
	imshow("����", im1);
	imwrite(_dst1_,im1);                       //���Ϊim1

	inputim = im;
	Mat im2;                               //ƽ������ֵ���˲�����
	blur(inputim, im2, Size(3,3));
	namedWindow("ƽ���˲�", 1);
	imshow("ƽ���˲�", im2);
	imwrite(_dst2_, im2);                  //���λim2


	inputim = im1;
	Mat im3;                               //��ֵ�˲�����
	medianBlur(inputim, im3, 3);
	namedWindow("��ֵ�˲�", 1);
	imshow("��ֵ�˲�", im3);
	imwrite(_dst3_, im3);                 //���ΪIm3

	waitKey(0);
	return 0;
}



