#include "stdafx.h"
#include "dps.h"
#include <random>

//CS����תCHAR
char* StrToChar(CString str)
{
	int n = str.GetLength();
	int len = WideCharToMultiByte(CP_ACP, 0, str, str.GetLength(), NULL, 0, NULL, NULL);
	char * src = new char[len + 1];   //���ֽ�Ϊ��λ
	WideCharToMultiByte(CP_ACP, 0, str, str.GetLength() + 1, src, len + 1, NULL, NULL);
	src[len] = '\0';               //���ֽ��ַ���'\0'����
	return src;
}
/*********************************************************************
*                           ���λ���������                           *
*********************************************************************/
Mat addSaltNoise(const Mat srcImage, int n,int m)
{
	Mat dstImage = srcImage.clone();
	srand(time(NULL));
	if (m == 0 || m == 2)
	{

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
	}

	if (m == 1 || m == 2)
	{

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
	}
	return dstImage;
};



/*********************************************************************
*                          ֱ��ͼ����                          *
*********************************************************************/

Mat Hisg(const Mat Mt)
{
	Mat M = Mt.clone();
	    int Max = 0;
		Mat Mhis(300, 256 * 2, CV_8UC1);
		int amount[256 * 2] = { 0 };
		int p = 0;
		for (int i = 0;i < Mhis.rows;i++)        //��ʼ��Ϊ��ɫ
		{
			for (int j = 0;j < Mhis.cols;j++)
			{
				Mhis.at<uchar>(i, j) = 255;
			}
		}
		for (int i = 0;i < M.rows;i++)        //��ⲻͬ���صĸ���
		{
			for (int j = 0;j < M.cols;j++)
			{
				p = M.at<uchar>(i, j);
				amount[2 * p]++;
			}
		}
		for (p = 0;p < 256;p++)
		{
			if (amount[2 * p] > Max)
				Max = amount[2 * (p)];
		}
		for (p = 0;p < Mhis.cols;p++)    //���ֱ��ͼ
		{
			amount[p] = amount[p] / ((Max/300)+1);
			//if (amount[p] > 299)
			//	amount[p] = 299;
			for (int i = Mhis.rows - amount[p] - 1;i < Mhis.rows;i++)      //���ֱ��ͼ
			{
				Mhis.at<uchar>(i, p) = 0;
			}

		}

	//else
	//{
	//	Mat Mhis(300, 256 * 2, CV_8UC1);
	//	int amount[256 * 2] = { 0 };
	//	int p = 0;
	//	for (int i = 0;i < Mhis.rows;i++)        //��ʼ��Ϊ��ɫ
	//	{
	//		for (int j = 0;j < Mhis.cols;j++)
	//		{
	//			Mhis.at<uchar>(i, j) = 255;
	//		}
	//	}
	//	for (int i = 0;i < M.rows;i++)        //��ⲻͬ���صĸ���
	//	{
	//		for (int j = 0;j < M.cols;j++)
	//		{
	//			p = M.at<uchar>(i, j);
	//			amount[2 * p]++;
	//		}
	//	}
	//	for (p = 0;p < Mhis.cols;p++)    //���ֱ��ͼ
	//	{
	//		amount[p] = amount[p] / 10;
	//		if (amount[p] > 299)
	//			amount[p] = 299;
	//		for (int i = Mhis.rows - amount[p] - 1;i < Mhis.rows;i++)      //���ֱ��ͼ
	//		{
	//			Mhis.at<uchar>(i, p) = 0;
	//		}

	//	}
	//}

	return Mhis;
}

/*********************************************************************
*                         ���⻯/��һ������                          *
*********************************************************************/

Mat Eqlz(const Mat M,int Range[])
{
	Mat Me = M.clone();
	int amount[256] = { 0 };
	int p = 0;
	int sum[256] = { 0 };

		for (int i = 0;i<Me.rows;i++)        //��ⲻͬ���صĸ���
		{
			for (int j = 0;j<Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
					amount[p]++;
			}
		}
		sum[0] = amount[0];
		for (int k = 1;k < 256;k++)
		{
			sum[k] = amount[k]+sum[k-1];
		}

		for (int i = 0;i<Me.rows;i++)        //��ⲻͬ���صĸ���
		{
			for (int j = 0;j<Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
				Me.at<uchar>(i, j) = Range[0]+(Range[1]-Range[0])* sum[p] / (Me.rows*Me.cols);
			}
		}





	return Me;
}

/*********************************************************************
*                          �Ҷȷֲ㺯��                           *
*********************************************************************/

Mat Glvls(const Mat M,int Rang[],int Mark)
{
	Mat Me = M.clone();
	int amount[256] = { 0 };
	int p = 0;
	if (Mark==1)
	{

		for (int i = 0;i < Me.rows;i++)        //��ⲻͬ���صĸ���
		{
			for (int j = 0;j < Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
				if (p >= Rang[0] && p <= Rang[1])
					Me.at<uchar>(i, j) = 200;



			}
		}
	}
	else
	{

		for (int i = 0;i < Me.rows;i++)        //��ⲻͬ���صĸ���
		{
			for (int j = 0;j < Me.cols;j++)
			{
				p = Me.at<uchar>(i, j);
				if (p >= Rang[0] && p <= Rang[1])
					Me.at<uchar>(i, j) = 155;
				else
					Me.at<uchar>(i, j) = 30;



			}
		}
	}



	return Me;
}

/*********************************************************************
*                          ٤��任����                           *
*********************************************************************/

Mat Gamma_ma(const Mat M,double y)
{
	Mat Me = M.clone();
	int p = 0;
	double c=255./ pow(255, y);

	for (int i = 0;i<Me.rows;i++)        //��ⲻͬ���صĸ���
	{
		for (int j = 0;j<Me.cols;j++)
		{
			p = Me.at<uchar>(i, j);
			Me.at<uchar>(i, j) = (uchar)(c* pow(p, y));


		}
	}


	return Me;
}

/*********************************************************************
*                         DFT��IDFT                     *
*********************************************************************/
void Dft_IDft(const Mat M)
{
	Mat inputim = M.clone();
	int w = getOptimalDFTSize(inputim.cols);
	int h = getOptimalDFTSize(inputim.rows);//��ȡ��ѳߴ磬���ٸ���Ҷ�任Ҫ��ߴ�Ϊ2��n�η�  
	Mat padded;
	copyMakeBorder(inputim, padded, 0, h - inputim.rows, 0, w - inputim.cols, BORDER_CONSTANT, Scalar::all(0));//���ͼ�񱣴浽padded��  
	Mat plane[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };//����ͨ��  
	Mat complexIm;
	merge(plane, 2, complexIm);//�ϲ�ͨ��  
	dft(complexIm, complexIm);//���и���Ҷ�任���������������  
	split(complexIm, plane);//����ͨ��  
	magnitude(plane[0], plane[1], plane[0]);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ��������Ϊ��ά����Ҷ�任����Ǹ���  
	int cx = padded.cols / 2;int cy = padded.rows / 2;//һ�µĲ������ƶ�ͼ�����������½���λ�ã����������½���λ��  
	Mat temp;
	Mat part1(plane[0], Rect(0, 0, cx, cy));
	Mat part2(plane[0], Rect(cx, 0, cx, cy));
	Mat part3(plane[0], Rect(0, cy, cx, cy));
	Mat part4(plane[0], Rect(cx, cy, cx, cy));


	part1.copyTo(temp);
	part4.copyTo(part1);
	temp.copyTo(part4);

	part2.copyTo(temp);
	part3.copyTo(part2);
	temp.copyTo(part3);
	//******************************************************************* 
	Mat &mag = plane[0];


	//Mat _complexim(complexIm,Rect(padded.cols/4,padded.rows/4,padded.cols/2,padded.rows/2));  
	//opyMakeBorder(_complexim,_complexim,padded.rows/4,padded.rows/4,padded.cols/4,padded.cols/4,BORDER_CONSTANT,Scalar::all(0.75));  
	Mat _complexim;
	complexIm.copyTo(_complexim);//�ѱ任�������һ�ݣ�������任��Ҳ���ǻָ�ԭͼ  
	Mat iDft[] = { Mat::zeros(plane[0].size(),CV_32F),Mat::zeros(plane[0].size(),CV_32F) };//��������ͨ��������Ϊfloat����СΪ����ĳߴ�  
	idft(_complexim, _complexim);//����Ҷ��任  
	split(_complexim, iDft);//���ò��Ҳ�Ǹ���  
	//magnitude(iDft[0], iDft[1], iDft[0]);//����ͨ������Ҫ��ȡ0ͨ��  
	normalize(iDft[0], iDft[0], 1, 0, CV_MINMAX);//��һ������float���͵���ʾ��ΧΪ0-1,����1Ϊ��ɫ��С��0Ϊ��ɫ  
	imshow("idft", iDft[0]);//��ʾ��任  
							//*******************************************************************  
	plane[0] += Scalar::all(1);//����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�  
	log(plane[0], plane[0]);
	normalize(plane[0], plane[0], 1, 0, CV_MINMAX);

	imshow("dft", plane[0]);
}

/*********************************************************************
*                          �����ͨ�˲���                     *
*********************************************************************/

Mat ideal_Low_Pass_Filter(Mat src,int D0) 
{
	Mat img=src.clone();

	//����ͼ����ٸ���Ҷ�任  
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	//��¼����Ҷ�任��ʵ�����鲿  
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	//���и���Ҷ�任  
	dft(complexImg, complexImg);
	//��ȡͼ��  
	Mat &mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));//����Ϊʲô&��-2����鿴opencv�ĵ�  
														//��ʵ��Ϊ�˰��к��б��ż�� -2�Ķ�������11111111.......10 ���һλ��0  
														//��ȡ���ĵ�����  
	int cx = padded.cols / 2;int cy = padded.rows / 2;//һ�µĲ������ƶ�ͼ�����������½���λ�ã����������½���λ��  
	//����Ƶ��  
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//DoΪ�Լ��趨�ķ�ֵ���忴��ʽ  
	//double D0 = 60;
	//������ʽ�������Ĳ���  
	for (int y = 0; y < mag.rows; y++) {
		double* data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			if (d <= D0) {

			}
			else {
				data[x] = 0;
			}
		}
	}

	split(mag, planes);//����ͨ��  
	magnitude(planes[0], planes[1], planes[0]);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ��������Ϊ��ά����Ҷ�任����Ǹ���  

	planes[0] += Scalar::all(1);//����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�  
	log(planes[0], planes[0]);
	normalize(planes[0], planes[0], 1, 0, CV_MINMAX);
	imshow("�˲���", planes[0]);





	//imshow("�˲���", M_Fil);
	//�ٵ���Ƶ��  
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//��任  
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT  
	invDFT.convertTo(invDFTcvt, CV_8U);
	return invDFTcvt;
}

/*********************************************************************
*                         ������˹��ͨ�˲���                     *
*********************************************************************/
Mat Butterworth_Low_Paass_Filter(Mat src,int D0,int n)
{
	/*int n = 1;*///��ʾ������˹�˲����Ĵ���  
			  //H = 1 / (1+(D/D0)^2n)  
	Mat img=src.clone();

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//double D0 = 100;

	for (int y = 0; y < mag.rows; y++) {
		float* data = mag.ptr<float>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			double s = (double)d / D0;
			double h = (double)1 / (1 + pow(s, 2 * n));

			//if (h <= 0.01)
			//{
			//	data[2*x] = 0;
			//	data[2*x+1] = 0;
			//}
			//else {
		        data[2*x] = data[2 * x]* h;
				data[2 * x + 1] = data[2 * x+1]*h;
			//}
		}
	}

	split(mag, planes);//����ͨ��  
	magnitude(planes[0], planes[1], planes[0]);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ��������Ϊ��ά����Ҷ�任����Ǹ���  

	planes[0] += Scalar::all(1);//����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�  
	log(planes[0], planes[0]);
	normalize(planes[0], planes[0], 1, 0, CV_MINMAX);
	imshow("�˲���", planes[0]);

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//��任  
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT  
	invDFT.convertTo(invDFTcvt, CV_8U);
	return invDFTcvt;
}

/*********************************************************************
*                         ��˹��ͨ�˲���                     *
*********************************************************************/
Mat Gauss_Low_Paass_Filter(Mat src, int sigma)
{
	/*int n = 1;*///��ʾ������˹�˲����Ĵ���  
				  //H = 1 / (1+(D/D0)^2n)  
	Mat img = src.clone();

	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);


	//static const double pi = 3.1415926;
	//int xcenter = padded.cols/2;
	//int ycenter = padded.rows / 2;
	//double **gauss = new double *[padded.cols];
	//for (int i = 0;i<padded.cols;i++)
	//	gauss[i] = new double[padded.rows];//Ϊÿ�з���ռ䣨ÿ������col��Ԫ�أ� 
	//double x2, y2;
	//double org = (double)1 / 2 * pi * sigma;
	//for (int i = 0; i < padded.cols; i++)
	//{
	//	x2 = pow(i - xcenter, 2);
	//	for (int j = 0; j < padded.rows; j++)
	//	{
	//		y2 = pow(j - ycenter, 2);
	//		double g = exp(-(x2 + y2) / (2 * sigma * sigma));
	//		g /= 2 * pi * sigma;
	//		gauss[i][j] = g/org;
	//	}
	//}


	for (int y = 0; y < mag.rows; y++) {
		float* data = mag.ptr<float>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			double s2 = pow(d,2) / (2*pow(sigma,2));
			double h = exp(-s2);

			//if (h <= 0.607)
			//{
			//	data[2*x] = 0;
			//	data[2*x+1] = 0;
			//}
			//else {
			data[2 * x] *= h;
			data[2 * x + 1] *= h;
			//}
		}
	}

	//for (int i = 0;i<padded.cols;i++)
	//	delete gauss[i];
	//delete[] gauss;

	split(mag, planes);//����ͨ��  
	magnitude(planes[0], planes[1], planes[0]);//��ȡ����ͼ��0ͨ��Ϊʵ��ͨ����1Ϊ��������Ϊ��ά����Ҷ�任����Ǹ���  

	planes[0] += Scalar::all(1);//����Ҷ�任���ͼƬ���÷��������ж�����������ȽϺÿ�  
	log(planes[0], planes[0]);
	normalize(planes[0], planes[0], 1, 0, CV_MINMAX);
	imshow("�˲���", planes[0]);

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//��任  
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT  
	invDFT.convertTo(invDFTcvt, CV_8U);
	return invDFTcvt;
}

/*********************************************************************
*                         ��˹����                  *
*********************************************************************/
double generateGaussianNoise(double mu, double sigma)
{
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846;
	thread_local double z1;
	thread_local bool generate;
	generate = !generate;

	if (!generate)
		return z1 * sigma + mu;

	double u1, u2;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

Mat GaussNoise(const Mat src, int sm)
{
	Mat im=src.clone();
	for (int i = 0 ;i < im.rows;i++)
	{
		uchar* Data = im.ptr<uchar>(i);
		for (int j = 0;j < im.cols;j++)
		{
			Data[j] = (uchar)generateGaussianNoise(Data[j], (double)sm);
			if (Data[j] < 0)
				Data[j] = 0;
			if (Data[j] > 255)
				Data[j] = 255;
		}
	}
	return im;

}

/*********************************************************************
*                         ��������                  *
*********************************************************************/

Mat RayleighNoise(const Mat src, int sm)
{
	Mat im = src.clone();
	double u = 0;
	srand(time(NULL));
	for (int i = 0;i < im.rows;i++)
	{
		uchar* Data = im.ptr<uchar>(i);
		for (int j = 0;j < im.cols;j++)
		{
			u = rand() * (1.0 / RAND_MAX);
			u = sqrt(-2 * log(u));
			u = sm * 0.655*u;
			Data[j] += (uchar)u;
			if (Data[j] < 0)
				Data[j] = 0;
			if (Data[j] > 255)
				Data[j] = 255;
		}
	}
	return im;

}

/*********************************************************************
*                         ָ���ֲ�����                  *
*********************************************************************/

Mat IndexNoise(const Mat src, double st)
{
	Mat im = src.clone();
	double u = 0;
	srand(time(NULL));
	for (int i = 0;i < im.rows;i++)
	{
		uchar* Data = im.ptr<uchar>(i);
		for (int j = 0;j < im.cols;j++)
		{
			u = rand() * (1.0 / RAND_MAX);
			u = -log(1 - u) / st;
			/*u = ((double)1/st) * u;*/
			Data[j] = (uchar)Data[j]+u;
			if (Data[j] < 0)
				Data[j] = 0;
			if (Data[j] > 255)
				Data[j] = 255;
		}
	}
	return im;

}
/*********************************************************************
*                         ٤��ֲ�����                  *
*********************************************************************/


Mat GammaNoise(const Mat src, double aph,double lda)
{
	Mat im = src.clone();
	double u = 0;
	std::default_random_engine generator(time(NULL));
	std::gamma_distribution<double> distribution(aph, lda);
	for (int i = 0;i < im.rows;i++)
	{
		uchar* Data = im.ptr<uchar>(i);
		for (int j = 0;j < im.cols;j++)
		{
			Data[j] = (uchar)Data[j] + (uchar)distribution(generator);
			if (Data[j] < 0)
				Data[j] = 0;
			if (Data[j] > 255)
				Data[j] = 255;
		}
	}
	return im;

}

/*********************************************************************
*                         ���ȷֲ�����                  *
*********************************************************************/

Mat UniformNoise(const Mat src, double a)
{
	Mat im = src.clone();
	double u = 0;
	srand(time(NULL));
	for (int i = 0;i < im.rows;i++)
	{
		uchar* Data = im.ptr<uchar>(i);
		for (int j = 0;j < im.cols;j++)
		{
			u = rand() * (1.0 / RAND_MAX);
			u *= a;
			Data[j] = (uchar)Data[j] + u;
			if (Data[j] < 0)
				Data[j] = 0;
			if (Data[j] > 255)
				Data[j] = 255;
		}
	}
	return im;

}

/*********************************************************************
*                         ���ξ�ֵ�˲�                 *
*********************************************************************/
Mat GeometricMeanFilter(const Mat src, int n)
{
	Mat im(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	Mat dst(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	int x = (n - 1) / 2;
	int y = (n - 1) / 2;
	int Endx = im.cols -1- x;
	int Endy = im.rows -1- y;
	src.copyTo(im(Rect(x, y, src.cols, src.rows)));

	Mat up1 = im(Rect(x+1, 0, src.cols, y));
	Mat up0 = im(Rect(x+1, y+1, src.cols, y));
	Mat down1 = im(Rect(x+1, Endy + 1, src.cols, y));
	Mat down0 = im(Rect(x+1, Endy - y, src.cols, y));
	Mat left1 = im(Rect(0, y + 1, x, src.rows));
	Mat left0 = im(Rect(x + 1, y + 1, x, src.rows));
	Mat right1 = im(Rect(Endx + 1, y + 1, x, src.rows));
	Mat right0 = im(Rect(Endx - y, y + 1, x, src.rows));

	Mat lu1 = im(Rect(0, 0, x, y));
	Mat lu0 = im(Rect(x + 1, y + 1, x, y));
	Mat ld1 = im(Rect(0, Endy + 1, x, y));
	Mat ld0 = im(Rect(x + 1, Endy - y, x, y));
	Mat ru1 = im(Rect(Endx + 1, 0, x, y));
	Mat ru0 = im(Rect(Endx - x, y + 1, x, y));
	Mat rd1 = im(Rect(Endx + 1, Endy + 1, x, y));
	Mat rd0 = im(Rect(Endx - x, Endy - y, x, y));

	flip(up0, up1, 0);
	flip(down0, down1, 0);
	flip(right0, right1, 1);
	flip(left0, left1, 1);
	flip(lu0, lu1, -1);
	flip(ld0, ld1, -1);
	flip(ru0, ru1, -1);
	flip(rd0, rd1, -1);

	for (int j = y;j < Endy + 1;j++)
	{
		uchar* data = dst.ptr<uchar>(j);
		for (int i = x;i < Endx + 1;i++)
		{


			double product = 1;
			for (int m = j - y;m < j + y+1;m++)
			{
				uchar* data2 = im.ptr<uchar>(m);
				for (int n = i - x;n < i + x+1;n++)
				{
					if (data2[n]!= 0)
					product *= data2[n];

				}
			}

			data[i] = pow(product,1.0/(double)(n*n));



		}
	}
	dst(Rect(x, y, src.cols, src.rows)).copyTo(dst);
	return dst;
}

/*********************************************************************
*                         г����ֵ�˲�                 *
*********************************************************************/
Mat HarmonicMeanFilter(const Mat src, int n)
{
	Mat im(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	Mat dst(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	int x = (n - 1) / 2;
	int y = (n - 1) / 2;
	int Endx = im.cols - 1 - x;
	int Endy = im.rows - 1 - y;
	src.copyTo(im(Rect(x, y, src.cols, src.rows)));

	Mat up1 = im(Rect(x + 1, 0, src.cols, y));
	Mat up0 = im(Rect(x + 1, y + 1, src.cols, y));
	Mat down1 = im(Rect(x + 1, Endy + 1, src.cols, y));
	Mat down0 = im(Rect(x + 1, Endy - y, src.cols, y));
	Mat left1 = im(Rect(0, y + 1, x, src.rows));
	Mat left0 = im(Rect(x + 1, y + 1, x, src.rows));
	Mat right1 = im(Rect(Endx + 1, y + 1, x, src.rows));
	Mat right0 = im(Rect(Endx - y, y + 1, x, src.rows));

	Mat lu1 = im(Rect(0, 0, x, y));
	Mat lu0 = im(Rect(x + 1, y + 1, x, y));
	Mat ld1 = im(Rect(0, Endy + 1, x, y));
	Mat ld0 = im(Rect(x + 1, Endy - y, x, y));
	Mat ru1 = im(Rect(Endx + 1, 0, x, y));
	Mat ru0 = im(Rect(Endx - x, y + 1, x, y));
	Mat rd1 = im(Rect(Endx + 1, Endy + 1, x, y));
	Mat rd0 = im(Rect(Endx - x, Endy - y, x, y));

	flip(up0, up1, 0);
	flip(down0, down1, 0);
	flip(right0, right1, 1);
	flip(left0, left1, 1);
	flip(lu0, lu1, -1);
	flip(ld0, ld1, -1);
	flip(ru0, ru1, -1);
	flip(rd0, rd1, -1);

	for (int j = y;j < Endy + 1;j++)
	{
		uchar* data = dst.ptr<uchar>(j);
		for (int i = x;i < Endx + 1;i++)
		{


			double sum = 0;
			for (int m = j - y;m < j + y + 1;m++)
			{
				uchar* data2 = im.ptr<uchar>(m);
				for (int n = i - x;n < i + x + 1;n++)
				{
					if (data2[n] != 0)
						sum += 1.0/data2[n];

				}
			}

			data[i] = n*n / sum;



		}
	}
	dst(Rect(x, y, src.cols, src.rows)).copyTo(dst);
	return dst;
}

/*********************************************************************
*                         ��г����ֵ�˲�                 *
*********************************************************************/
Mat iHarmonicMeanFilter(const Mat src, int n,int q)
{
	Mat im(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	Mat dst(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	int x = (n - 1) / 2;
	int y = (n - 1) / 2;
	int Endx = im.cols - 1 - x;
	int Endy = im.rows - 1 - y;
	src.copyTo(im(Rect(x, y, src.cols, src.rows)));

	Mat up1 = im(Rect(x + 1, 0, src.cols, y));
	Mat up0 = im(Rect(x + 1, y + 1, src.cols, y));
	Mat down1 = im(Rect(x + 1, Endy + 1, src.cols, y));
	Mat down0 = im(Rect(x + 1, Endy - y, src.cols, y));
	Mat left1 = im(Rect(0, y + 1, x, src.rows));
	Mat left0 = im(Rect(x + 1, y + 1, x, src.rows));
	Mat right1 = im(Rect(Endx + 1, y + 1, x, src.rows));
	Mat right0 = im(Rect(Endx - y, y + 1, x, src.rows));

	Mat lu1 = im(Rect(0, 0, x, y));
	Mat lu0 = im(Rect(x + 1, y + 1, x, y));
	Mat ld1 = im(Rect(0, Endy + 1, x, y));
	Mat ld0 = im(Rect(x + 1, Endy - y, x, y));
	Mat ru1 = im(Rect(Endx + 1, 0, x, y));
	Mat ru0 = im(Rect(Endx - x, y + 1, x, y));
	Mat rd1 = im(Rect(Endx + 1, Endy + 1, x, y));
	Mat rd0 = im(Rect(Endx - x, Endy - y, x, y));

	flip(up0, up1, 0);
	flip(down0, down1, 0);
	flip(right0, right1, 1);
	flip(left0, left1, 1);
	flip(lu0, lu1, -1);
	flip(ld0, ld1, -1);
	flip(ru0, ru1, -1);
	flip(rd0, rd1, -1);

	for (int j = y;j < Endy + 1;j++)
	{
		uchar* data = dst.ptr<uchar>(j);
		for (int i = x;i < Endx + 1;i++)
		{


			double sum_n = 0;
			double sum_d = 0;
			for (int m = j - y;m < j + y + 1;m++)
			{
				uchar* data2 = im.ptr<uchar>(m);
				for (int n = i - x;n < i + x + 1;n++)
				{
					sum_n += pow(data2[n],q+1);
					sum_d += pow(data2[n], q);
				}
			}

			data[i] = sum_n / sum_d;

		}
	}
	dst(Rect(x, y, src.cols, src.rows)).copyTo(dst);
	return dst;
}

/*********************************************************************
*                         �����˲���                 *
*********************************************************************/
Mat OrderFilter(const Mat src, int n, int m)
{
	Mat im(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	Mat dst(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	int x = (n - 1) / 2;
	int y = (n - 1) / 2;
	int Endx = im.cols - 1 - x;
	int Endy = im.rows - 1 - y;
	src.copyTo(im(Rect(x, y, src.cols, src.rows)));

	Mat up1 = im(Rect(x + 1, 0, src.cols, y));
	Mat up0 = im(Rect(x + 1, y + 1, src.cols, y));
	Mat down1 = im(Rect(x + 1, Endy + 1, src.cols, y));
	Mat down0 = im(Rect(x + 1, Endy - y, src.cols, y));
	Mat left1 = im(Rect(0, y + 1, x, src.rows));
	Mat left0 = im(Rect(x + 1, y + 1, x, src.rows));
	Mat right1 = im(Rect(Endx + 1, y + 1, x, src.rows));
	Mat right0 = im(Rect(Endx - y, y + 1, x, src.rows));

	Mat lu1 = im(Rect(0, 0, x, y));
	Mat lu0 = im(Rect(x + 1, y + 1, x, y));
	Mat ld1 = im(Rect(0, Endy + 1, x, y));
	Mat ld0 = im(Rect(x + 1, Endy - y, x, y));
	Mat ru1 = im(Rect(Endx + 1, 0, x, y));
	Mat ru0 = im(Rect(Endx - x, y + 1, x, y));
	Mat rd1 = im(Rect(Endx + 1, Endy + 1, x, y));
	Mat rd0 = im(Rect(Endx - x, Endy - y, x, y));

	flip(up0, up1, 0);
	flip(down0, down1, 0);
	flip(right0, right1, 1);
	flip(left0, left1, 1);
	flip(lu0, lu1, -1);
	flip(ld0, ld1, -1);
	flip(ru0, ru1, -1);
	flip(rd0, rd1, -1);
	
	for (int j = y;j < Endy + 1;j++)
	{
		uchar* data = dst.ptr<uchar>(j);
		for (int i = x;i < Endx + 1;i++)
		{


			double sum_n = 0;
			double sum_d = 0;
			int Max  = im.ptr<uchar>(j - y)[i - x];
			int Min = Max;
			for (int m = j - y;m < j + y + 1;m++)
			{
				uchar* data2 = im.ptr<uchar>(m);
				for (int n = i - x;n < i + x + 1;n++)
				{
					if (data2[n] > Max)
						Max = data2[n];
					if (data2[n] < Min)
						Min = data2[n];

				}
			}
			switch (m)
			{
			case 0:data[i] = Min;break;
			case 1:data[i] = (Max + Min) / 2;break;
			case 2:data[i] = Max;break;
			}


		}
	}
	dst(Rect(x, y, src.cols, src.rows)).copyTo(dst);
	return dst;
}

/*********************************************************************
*                         �����İ������˲���                 *
*********************************************************************/
Mat AlphaFilter(const Mat src, int n, int d)
{
	Mat im(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	Mat dst(src.rows + n - 1, src.cols + n - 1, CV_8UC1, Scalar::all(0));
	int x = (n - 1) / 2;
	int y = (n - 1) / 2;
	int Endx = im.cols - 1 - x;
	int Endy = im.rows - 1 - y;
	src.copyTo(im(Rect(x, y, src.cols, src.rows)));

	Mat up1 = im(Rect(x + 1, 0, src.cols, y));
	Mat up0 = im(Rect(x + 1, y + 1, src.cols, y));
	Mat down1 = im(Rect(x + 1, Endy + 1, src.cols, y));
	Mat down0 = im(Rect(x + 1, Endy - y, src.cols, y));
	Mat left1 = im(Rect(0, y + 1, x, src.rows));
	Mat left0 = im(Rect(x + 1, y + 1, x, src.rows));
	Mat right1 = im(Rect(Endx + 1, y + 1, x, src.rows));
	Mat right0 = im(Rect(Endx - y, y + 1, x, src.rows));

	Mat lu1 = im(Rect(0, 0, x, y));
	Mat lu0 = im(Rect(x + 1, y + 1, x, y));
	Mat ld1 = im(Rect(0, Endy + 1, x, y));
	Mat ld0 = im(Rect(x + 1, Endy - y, x, y));
	Mat ru1 = im(Rect(Endx + 1, 0, x, y));
	Mat ru0 = im(Rect(Endx - x, y + 1, x, y));
	Mat rd1 = im(Rect(Endx + 1, Endy + 1, x, y));
	Mat rd0 = im(Rect(Endx - x, Endy - y, x, y));

	flip(up0, up1, 0);
	flip(down0, down1, 0);
	flip(right0, right1, 1);
	flip(left0, left1, 1);
	flip(lu0, lu1, -1);
	flip(ld0, ld1, -1);
	flip(ru0, ru1, -1);
	flip(rd0, rd1, -1);

	uchar* order = new uchar[n*n];
	for (int j = y;j < Endy + 1;j++)
	{
		uchar* data = dst.ptr<uchar>(j);
		for (int i = x;i < Endx + 1;i++)
		{


			double sum = 0;
			int temp = 0;
			order[0] = im.ptr<uchar>(j - y)[i - x];
			int c = 0;
			for (int m = j - y;m < j + y + 1;m++)
			{
				uchar* data2 = im.ptr<uchar>(m);
				for (int n = i - x;n < i + x + 1;n++)
				{
					
					order[c++] = data2[n];
					sum += data2[n];
				}
			}
			
			for (int a = 0;a < c-1;a++)
			{
				int flag = 1;
				int n = a;
				while (order[n] < order[n + 1])
				{
					temp = order[n];
					order[n] = order[n+1];
					order[n + 1] = temp;
					n--;
					if (n < 0)
					n = 0;
				}
			}
			
			for (int a = 0;a < (d / 2);a++)
			{
				sum -= order[a];
				sum -= order[c - 1 - a];
			}

			data[i] = sum / (n*n - d);



		}
	}
	delete order;

	dst(Rect(x, y, src.cols, src.rows)).copyTo(dst);
	return dst;
}

/*********************************************************************
*                        ����Ӧ�˲�                 *
*********************************************************************/
Mat SelfAdaptedFilter(const Mat src, int n,double vari_n)
{
	Mat im = src.clone();
	int M = im.rows%n;
	int N = im.cols%n;
	if (M != 0)M = n - M;
	if (N != 0)N = n - N;
	copyMakeBorder(im, im, 0, M , 0, N , BORDER_DEFAULT);


	uchar* num = new uchar[n*n];
	for (int j = 0;j <im.rows/n;j++)
	{
		for (int i = 0;i <im.cols/n ;i++)
		{


			double sum = 0;
			double sum_v = 0;
			int aver = 0;
			double vari_l = 0;
			int c = 0;
			for (int m = j*n;m < j*n+n;m++)
			{
				uchar* data = im.ptr<uchar>(m);
				for (int t = i*n;t < i*n+n;t++)
				{
					sum += data[t];
					num[c++] = data[t];
				}
			}

			aver = sum / c;
			for (int a = 0;a < c;a++)
			{
				sum_v += pow(num[a] - aver, 2);
			}
			vari_l = sum_v / c;

			double h = vari_n / vari_l;
			if (vari_n > vari_l)
				h = 1;
			for (int m = j * n;m < j*n + n;m++)
			{
				uchar* data = im.ptr<uchar>(m);
				for (int t = i * n;t < i*n + n;t++)
				{
					data[t] = data[t] - h * (data[t] - aver);
				}
			}




		}
	}
	im(Rect(0, 0, src.cols, src.rows)).copyTo(im);
	delete num;
	return im;
}