
#include <opencv.hpp>
#include <iostream> 
#include <time.h>
using namespace std;
using namespace cv;


#define _dst1_ "C:\\Users\\Cashion\\Desktop\\something\\object1.jpg"    //����ͼ��·��1
#define _dst2_ "C:\\Users\\Cashion\\Desktop\\something\\object2.jpg"    //����ͼ��·��2
#define _dst3_ "C:\\Users\\Cashion\\Desktop\\something\\object3.jpg"    //����ͼ��·��3

Mat addSaltNoise(const Mat srcImage, int n);   //���λ�������������
Mat Hisg(const Mat M);
Mat Eqlz(const Mat M, int Range[]);
Mat Glvls(const Mat M);
