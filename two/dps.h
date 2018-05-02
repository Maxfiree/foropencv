
#include <opencv.hpp>
#include <iostream> 
#include <time.h>
using namespace std;
using namespace cv;


#define _dst1_ "C:\\Users\\Cashion\\Desktop\\something\\object1.jpg"    //����ͼ��·��1
#define _dst2_ "C:\\Users\\Cashion\\Desktop\\something\\object2.jpg"    //����ͼ��·��2
#define _dst3_ "C:\\Users\\Cashion\\Desktop\\something\\object3.jpg"    //����ͼ��·��3

Mat addSaltNoise(const Mat srcImage, int n,int m);   //���λ�������������
Mat Hisg(const Mat M);
Mat Eqlz(const Mat M, int Range[]);
Mat Glvls(const Mat M,int Rang[], int Mark);
char* StrToChar(CString str);
void Dft_IDft(const Mat Mt);
Mat Gamma_ma(const Mat M, double y);
Mat ideal_Low_Pass_Filter(Mat src,int D0);
Mat Butterworth_Low_Paass_Filter(Mat src, int D0, int n);
Mat Gauss_Low_Paass_Filter(Mat src, int sigma);
Mat GaussNoise(const Mat src, int sm);
Mat RayleighNoise(const Mat src, int sm);
Mat IndexNoise(const Mat src, double st);
Mat GammaNoise(const Mat src, double aph, double lda);
Mat UniformNoise(const Mat src, double a);
Mat GeometricMeanFilter(const Mat src, int n);
Mat HarmonicMeanFilter(const Mat src, int n);
Mat iHarmonicMeanFilter(const Mat src, int n, int q);
Mat OrderFilter(const Mat src, int n, int m);
Mat AlphaFilter(const Mat src, int n, int d);
Mat SelfAdaptedFilter(const Mat src, int n, double vari_n);