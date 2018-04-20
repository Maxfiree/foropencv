
#include <opencv.hpp>
#include <iostream> 
#include <time.h>
using namespace std;
using namespace cv;


#define _dst1_ "C:\\Users\\Cashion\\Desktop\\something\\object1.jpg"    //生成图像路径1
#define _dst2_ "C:\\Users\\Cashion\\Desktop\\something\\object2.jpg"    //生成图像路径2
#define _dst3_ "C:\\Users\\Cashion\\Desktop\\something\\object3.jpg"    //生成图像路径3

Mat addSaltNoise(const Mat srcImage, int n);   //椒盐化噪声函数声明
Mat Hisg(const Mat M);
Mat Eqlz(const Mat M, int Range[]);
Mat Glvls(const Mat M,int Rang[], int Mark);
char* StrToChar(CString str);
void Dft_IDft(const Mat Mt);
Mat Gamma_ma(const Mat M, double y);
Mat ideal_Low_Pass_Filter(Mat src,int D0);
Mat Butterworth_Low_Paass_Filter(Mat src, int D0, int n);
Mat Gauss_Low_Paass_Filter(Mat src, int sigma);